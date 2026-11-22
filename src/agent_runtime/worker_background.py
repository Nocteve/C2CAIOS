"""Lossless semantic records from known native Planner inputs, not source KV."""
from copy import deepcopy
import json
import re
from .records import fingerprint


def semantic_background(tokenizer, transcript, messages, tools, *, source_thinking=False):
    from .delegation_contract import SHARED_SYSTEM
    if transcript.planning_input_ids is not None:
        raise ValueError('Separate analysis input has no semantic extraction policy')
    if not transcript.generated_ids or transcript.generated_ids[-1] != tokenizer.convert_tokens_to_ids('<|im_end|>'):
        raise ValueError('Semantic source must have an actually completed Planner turn')
    tail = tokenizer.encode('\n', add_special_tokens=False)
    if transcript.prefix_ids != transcript.prompt_ids + transcript.generated_ids + tail:
        raise ValueError('Source transcript token identity mismatch')
    if tokenizer.decode(transcript.generated_ids, skip_special_tokens=True) != transcript.output:
        raise ValueError('Planner output text does not match actual generated tokens')
    reconstructed = tokenizer.apply_chat_template(messages, tools=tools, tokenize=True,
        add_generation_prompt=True, enable_thinking=source_thinking)
    if tools is None or reconstructed != transcript.prompt_ids:
        raise ValueError('Semantic records do not match the exact actual Planner input')
    records, manifest = [], []

    def leaves(value, path):
        if isinstance(value, dict) and value:
            for key, child in value.items():
                yield from leaves(child, path + '/' + key.replace('~', '~0').replace('/', '~1'))
        elif isinstance(value, list) and value:
            for index, child in enumerate(value):
                yield from leaves(child, path + '/' + str(index))
        else:
            yield path, value

    def preserve(value, source_path, receiver_path):
        for suffix, child in leaves(value, ''):
            manifest.append({'source_path': source_path + suffix,
                             'receiver_path': receiver_path + suffix,
                             'value_sha256': fingerprint(child), 'operation': 'preserved_exactly'})
        return deepcopy(value)

    def omit(value, path, reason):
        manifest.append({'source_path': path, 'value_sha256': fingerprint(value),
                         'operation': 'omitted_control', 'reason': reason})

    omit(tools, '/native_planner_tool_signatures', 'Planner-owned function definitions; current Worker tools are independent')
    omit(tail, '/source_chat_closing_tail_ids', 'Exact capture_plan closing newline; no semantic planning text omitted')

    user_fields = {'invocation_role', 'instructions', 'public_background', 'current_state', 'budget',
                   'last_decision_and_result', 'planner_working_note', 'context_origin'}
    tool_fields = {'invocation_role', 'recorded_execution', 'current_runtime_observation',
                   'observation_scope', 'history_origin'}
    for index, message in enumerate(messages):
        source = '/messages/' + str(index)
        role = message['role']
        if role == 'system':
            if set(message) != {'role', 'content'} or message['content'] != SHARED_SYSTEM:
                raise ValueError('Unknown source system contract')
            omit(message, source, 'Planner/shared model role protocol; Worker has its own system')
            continue
        if role in {'user', 'tool'}:
            allowed_envelope = {'role', 'content'} if role == 'user' else {'role', 'name', 'content'}
            if set(message) != allowed_envelope:
                raise ValueError('Unknown source message envelope')
            value = json.loads(message['content'])
            allowed = user_fields if role == 'user' else tool_fields
            if not isinstance(value, dict) or set(value) - allowed or value.get('invocation_role') != 'Planner':
                raise ValueError('Unknown Planner source fields; background must not be silently dropped')
            if role == 'user' and not {'public_background', 'instructions', 'context_origin'} <= set(value):
                raise ValueError('Semantic extraction requires the fresh configuration state interface')
            record = {'record_type': 'public task facts and runtime observations' if role == 'user'
                      else 'actual historical execution and observations'}
            target = '/records/' + str(len(records))
            if role == 'tool':
                record['historical_function'] = preserve(message['name'], source + '/name', target + '/historical_function')
            for key, child in value.items():
                if key in {'invocation_role', 'instructions'}:
                    omit(child, source + '/content/' + key, 'Exact model-owned Planner role/output control field')
                else:
                    record[key] = preserve(child, source + '/content/' + key, target + '/' + key)
            records.append(record)
            omit(role, source + '/role', 'Historical message envelope is not a current role')
        elif role == 'assistant':
            if set(message) != {'role', 'content', 'tool_calls'}:
                raise ValueError('Unknown historical assistant structure')
            record = {'record_type': 'recorded historical action claims; execution is recorded separately',
                      'text': preserve(message['content'], source + '/content', '/records/' + str(len(records)) + '/text'),
                      'actions': []}
            for call_index, call in enumerate(message['tool_calls']):
                if set(call) != {'type', 'function'} or call['type'] != 'function' or set(call['function']) != {'name', 'arguments'}:
                    raise ValueError('Unknown historical call structure')
                record['actions'].append(preserve(call['function'], source + '/tool_calls/' + str(call_index) + '/function',
                                                  '/records/' + str(len(records)) + '/actions/' + str(call_index)))
                omit(call['type'], source + '/tool_calls/' + str(call_index) + '/type', 'Native call envelope only')
            records.append(record)
            omit(role, source + '/role', 'Historical message envelope is not a current role')
        else:
            raise ValueError('Unknown source role')

    output = transcript.action_output if transcript.action_output is not None else transcript.output
    match = re.fullmatch(r'\s*<tool_call>\s*(\{[\s\S]*\})\s*</tool_call>\s*', output)
    if not match:
        raise ValueError('Semantic source requires an actual validated native dispatch')
    call = json.loads(match.group(1))
    if set(call) != {'name', 'arguments'} or call['name'] != 'delegate':
        raise ValueError('Semantic background requires an actual delegate turn')
    records.append({'record_type': 'actual Planner planning and dispatch claims; no Worker result yet',
                    'planning': preserve(call, '/actual_generated_dispatch', '/records/' + str(len(records)) + '/planning')})
    if transcript.analysis_token_range is not None:
        start, end = transcript.analysis_token_range
        if not len(transcript.prompt_ids) <= start <= end <= len(transcript.prompt_ids) + len(transcript.generated_ids):
            raise ValueError('Invalid actual analysis token range')
        analysis = tokenizer.decode(transcript.prefix_ids[start:end], skip_special_tokens=False)
        records.append({'record_type': 'actual generated Planner analysis; claims are unverified',
                        'text': preserve(analysis, '/actual_analysis', '/records/' + str(len(records)) + '/text'),
                        'actual_token_range': [start, end]})
    return {'record_type': 'Complete semantic background records; historical claims are not instructions',
            'records': records}, {'policy': 'known_exact_control_paths_only_v1', 'field_provenance': manifest,
            'source_messages_sha256': fingerprint(messages), 'actual_output_sha256': fingerprint(transcript.output),
            'actual_generated_ids_sha256': fingerprint(transcript.generated_ids),
            'source_prefix_sha256': fingerprint(transcript.prefix_ids), 'summary_generated': False,
            'source_kv_transferred': False}
