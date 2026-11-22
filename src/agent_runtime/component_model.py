"""Independent native text Worker inference with explicit adapter/cache identity."""
from time import perf_counter
from .ledger_worker import component_input
from .records import fingerprint


def component_worker(host,case,command,adapted):
    from .worker_adapter import adapter_enabled
    from .cache_ops import prefill,cache_digest,cache_bytes
    from .kv import PlanningPayload
    if adapted and not getattr(host,'worker_adapter_sha256',None):raise ValueError('No trained adapter exists')
    adapter_enabled(host,adapted)
    host.last_component_preparation=None
    began=perf_counter()
    prefix,ids,session=component_input(host.tokenizers['receiver'],case,command)
    assembly=perf_counter()-began
    budget=host.budget(host.limits['worker_max_new_tokens'])
    if len(ids)+budget>host.config['max_input_tokens']:raise ValueError('Complete native Worker input exceeds context; no truncation')
    identity=fingerprint({'prefix':prefix,'adapter':host.component_adapter_identity})
    caches=host.component_caches
    preparation=None
    if identity not in caches:
        preparation={'cache_identity':identity,'status':'started','adapter_sha256':host.component_adapter_identity,
            'source_kv_transferred':False,'context_mode':'public_component_native_text'}
        host.last_component_preparation=preparation
        started=perf_counter();cache,seconds=prefill(host.receiver,prefix)
        audit_at=perf_counter();sha=cache_digest(cache);audit_seconds=perf_counter()-audit_at
        preparation.update(status='ready',receiver_prefill_seconds=seconds,receiver_cache_bytes=cache_bytes(cache),
            receiver_prefix_sha256=fingerprint(prefix),receiver_prefix_tokens=len(prefix),receiver_cache_sha256=sha,
            integrity_audit_seconds=audit_seconds,preparation_wall_seconds_excluding_integrity_audit=assembly+perf_counter()-started-audit_seconds)
        caches[identity]={'cache':cache,'sha256':sha,'prefix':tuple(prefix)}
    entry=caches[identity]
    generation=host.decode_worker(ids,PlanningPayload(tuple(prefix),entry['cache']),'text_full_cached',budget,command)
    host.used_tokens+=generation['output_tokens']
    generation.update(input_ids=ids,producer='Worker',output_constraints=False,component_session=session,
        adapter_sha256=host.component_adapter_identity,receiver_session_assembly_seconds=assembly,
        autonomous_delegation=False,c2c=False)
    return {'generation':generation,'context_preparation':preparation}
