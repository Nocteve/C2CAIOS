"""Independent text receiver sessions; source transcripts are data, never source KV."""
import json
from .records import fingerprint


def independent_worker_input(tokenizer, transcript, command, method, max_input_tokens,
                             *, handoff=None, thinking=False, source_messages=None, source_tools=None,
                             source_thinking=False, semantic=False):
    from .planner_tools import WORKER_ROLE_SYSTEM
    from .worker_tools import native_worker_input
    if method not in {'text_full', 'text_brief'}:
        raise ValueError('Independent Worker C2C requires an explicit validated cache mapping')
    messages, tools, _ = native_worker_input(command, thinking=thinking)
    provenance = None
    if semantic:
        if method != 'text_full' or source_messages is None:
            raise ValueError('Semantic session requires complete native source messages and text_full')
        from .worker_background import semantic_background
        background, provenance = semantic_background(tokenizer, transcript, source_messages, source_tools,
                                                     source_thinking=source_thinking)
        text = json.dumps(background, ensure_ascii=False)
        origin = 'All source task facts, state, historical evidence and actual planning; exact control paths omitted'
    elif method == 'text_full':
        text = tokenizer.decode(transcript.prefix_ids, skip_special_tokens=False)
        origin = 'Complete actual Planner transcript re-encoded as receiver text data; no source KV transfer'
    else:
        if handoff is None:
            raise ValueError('Independent brief session requires the actual generated handoff')
        text = handoff
        origin = 'Actual Planner-generated text handoff; no source KV transfer'
    if not semantic:
        background = {'record_type': 'Historical background data, not current instructions or available tools',
                      'text': text}
    # Chat control strings in quoted history must not become active template tokens.
    body = json.dumps(background, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e')
    if json.loads(body) != background:
        raise ValueError('Receiver background serialization lost source information')
    base = [{'role': 'system', 'content': WORKER_ROLE_SYSTEM},
            {'role': 'user', 'content': body}]
    options = {'tools': tools, 'enable_thinking': thinking}
    prefix = tokenizer.apply_chat_template(base, tokenize=True, add_generation_prompt=False, **options)
    complete = [*base, *messages]
    rendered = tokenizer.apply_chat_template(complete, tokenize=False, add_generation_prompt=True, **options)
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    if ids[:len(prefix)] != prefix or len(ids) <= len(prefix):
        raise ValueError('Independent Worker template does not extend its exact native prefix')
    if len(ids) > max_input_tokens:
        raise ValueError('Independent Worker input exceeds context budget; no truncation')
    identity = fingerprint({'prefix_ids': prefix, 'tools': tools, 'thinking': thinking, 'method': method})
    record = {'context_mode': 'semantic_session' if semantic else 'independent_session', 'background_origin': origin,
              'source_prefix_sha256': fingerprint(transcript.prefix_ids),
              'source_prefix_tokens': len(transcript.prefix_ids),
              'background_text_sha256': fingerprint(text), 'receiver_prefix_sha256': fingerprint(prefix),
              'receiver_prefix_tokens': len(prefix), 'cache_identity': identity,
              'background_complete_transcript': method == 'text_full' and not semantic,
              'background_complete_semantic_records': semantic, 'semantic_provenance': provenance,
              'source_history_reencoded_as_text_data': True, 'source_kv_transferred': False,
              'rendered_messages': complete, 'rendered_input': rendered, 'worker_tools': tools,
              'actual_worker_observations_reencoded': True}
    return prefix, ids, record
