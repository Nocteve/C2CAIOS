"""Worker-owned native tools and real tool feedback, never Planner actions."""
import json
from .delegation_contract import object_schema
from .job_workspace import FIELDS
from .planner_tools import parse_native_action


def worker_schema(files, *, terminal_only=False, fields=FIELDS):
    text={"type":"string","minLength":1}
    edit=object_schema({"file":{"type":"string","enum":list(files)},
        "field":{"type":"string","enum":list(fields)},"value":{"type":"integer","minimum":0}})
    changes={"type":"array","items":edit,"maxItems":20}
    def action(name,**props):return object_schema({"action":{"type":"string","enum":[name]},**props})
    terminals=[action('report',summary=text,changes=changes),action('need_info',summary=text)]
    return {"anyOf":terminals if terminal_only else [action('inspect'),action('check',changes=changes),*terminals]}


def native_worker_input(command, *, thinking=False, fields=FIELDS):
    from copy import deepcopy
    payload=json.loads(command)
    if payload.get('invocation_role')!='Worker':raise ValueError('Expected an explicit Worker command')
    observations=payload.pop('tool_observations')
    payload['instructions']=(
        'Execute only the assigned local task using the shared background, snapshot and dependencies. '
        'Call exactly one of your provided Worker functions. Use a native tool_call envelope. '
        'inspect reads your granted files; check evaluates your explicit proposed edits without applying them. '
        'report submits your actual finding and explicit edits for Planner review; it does not modify files. '
        'Use an empty changes list only if you propose no edits. need_info reports missing information. '
        'Earlier Planner tools are unavailable to you. Do not claim work has been committed.')
    if thinking:
        payload['instructions']+=' Use native thinking before the final response; tool-call format restrictions apply to the final response after </think>.'
    remaining=payload['call_budget']['remaining_calls_including_this']
    if type(remaining) is not int or remaining<1:raise ValueError('Invalid remaining Worker call budget')
    schema=worker_schema(payload['granted_files'],terminal_only=remaining==1,fields=fields)
    if remaining==1:
        payload['instructions']+=' This is your last model call; return your actual report or missing information. There is no further tool round.'
    descriptions={'inspect':'Read the exact granted snapshot files.',
        'check':'Evaluate explicit candidate edits against the public rules without committing.',
        'report':'Return the actual local findings and explicit proposed edits to the Planner.',
        'need_info':'Report the concrete information missing for this assignment.'}
    tools=[]
    for variant in schema['anyOf']:
        parameters=deepcopy(variant);name=parameters['properties'].pop('action')['enum'][0]
        parameters['required'].remove('action')
        tools.append({'type':'function','function':{'name':name,'description':descriptions[name],'parameters':parameters}})
    messages=[{'role':'user','content':json.dumps(payload,ensure_ascii=False)}]
    for observation in observations:
        action=observation['worker_action']
        if action['action'] not in {'inspect','check'}:
            raise ValueError('Only executed nonterminal Worker calls have tool feedback')
        messages.append({'role':'assistant','content':'','tool_calls':[{'type':'function','function':{
            'name':action['action'],'arguments':{k:v for k,v in action.items() if k!='action'}}}]})
        messages.append({'role':'tool','name':action['action'],
            'content':json.dumps(observation['tool_result'],ensure_ascii=False)})
    return messages,tools,schema


def parse_native_worker(generation,task, *, remaining_calls):
    if generation.get('error') or generation.get('stop_reason')!='eos':
        raise ValueError(generation.get('error') or 'Incomplete Worker call')
    text=generation['action_output'] if generation.get('native_thinking') else generation['output']
    value=parse_native_action(text,worker_schema(task['files'],terminal_only=remaining_calls==1))
    # Reuse the original field, duplicate-edit and granted-scope validation.
    from .delegation_contract import parse_worker
    parse_worker({'output':json.dumps(value),'stop_reason':'eos'},task)
    return value,{'original_contract_passed':True,'transport_compatibility_used':False,
                  'interface':'native_worker_tools','semantic_repair':False}
