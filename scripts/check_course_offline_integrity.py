"""Check real preserved sources and actual recorded hidden-cache provenance.

Offline consistency, not a new GPU/tensor numerical replay.
"""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    baseline=ROOT/'artifacts/runtime-takeover-20261018-start'
    allowed={'src/agent_runtime/c2c_core.py','src/agent_runtime/model_service.py','src/agent_runtime/autonomy.py','scripts/run.py'}
    same=[];changed=[];unexpected=[]
    for folder in ['src','scripts','configs']:
        for old in (baseline/folder).rglob('*'):
            if not old.is_file() or '__pycache__' in old.parts:continue
            rel=old.relative_to(baseline).as_posix();now=ROOT/rel
            if now.exists() and sha(old)==sha(now):same.append(rel)
            elif rel in allowed:changed.append(rel)
            else:unexpected.append(rel)
    old_ast=ast.parse((baseline/'src/agent_runtime/c2c_core.py').read_text(encoding='utf-8'))
    now_ast=ast.parse((ROOT/'src/agent_runtime/c2c_core.py').read_text(encoding='utf-8'))
    old_fn=next(x for x in old_ast.body if isinstance(x,ast.FunctionDef)and x.name=='_run_hidden_arm')
    current_fn=next(x for x in now_ast.body if isinstance(x,ast.FunctionDef)and x.name=='_run_hidden_arm')
    defaults={a.arg:ast.unparse(v)for a,v in zip(current_fn.args.kwonlyargs,current_fn.args.kw_defaults)}
    class RestoreDefaults(ast.NodeTransformer):
        def visit_Name(self,node):
            if node.id=='role_system':return ast.Name(id='WORKER_ROLE_SYSTEM',ctx=ast.Load())
            if node.id=='max_new_tokens':return ast.Constant(value=320)
            return node
    normalized=RestoreDefaults().visit(deepcopy(current_fn));normalized.args=deepcopy(old_fn.args)
    default_body_same=ast.dump(normalized,include_attributes=False)==ast.dump(old_fn,include_attributes=False)
    normalized_module=deepcopy(now_ast)
    normalized_module.body=[deepcopy(old_fn)if isinstance(x,ast.FunctionDef)and x.name=='_run_hidden_arm'else x for x in normalized_module.body]
    rest_same=ast.dump(normalized_module,include_attributes=False)==ast.dump(old_ast,include_attributes=False)
    events_path=ROOT/'artifacts/correction-20261019/repair-comparison-v1/c2c_hidden_analysis/events.jsonl'
    publications={};calls=[]
    for line in events_path.read_text(encoding='utf-8').splitlines():
        event=json.loads(line)
        if event.get('type')!='model_response':continue
        response=event.get('response',{});result=response.get('result')or{};g=result.get('generation')or{};case=event.get('case_id')
        pub=result.get('published_version')
        if response.get('kind')=='planner' and pub:
            publications[(case,pub['id'])]=dict(metadata=pub,generation=g)
        if response.get('kind')=='worker' and g.get('published_cache_fused')and g.get('output_tokens',0)>0:
            pub=publications[(case,g['context_version'])];source=pub['generation'];lo,hi=source['analysis_token_range']
            selected=g['hidden_selection_source_indices']
            checks=dict(actual_analysis_positions=selected==list(range(lo,hi)),
                exact_real_reference_token_identity=g['hidden_reference_text_ids']==[source['prefix_ids'][i]for i in selected],
                source_cache_identity=g['source_cache_sha256']==pub['metadata']['source_cache_sha256'],
                source_not_reencoded=g['source_cache_was_reencoded']is False,
                actual_generated_tokens=g['output_tokens']>0,
                real_receiver_cache_hash=bool(g.get('receiver_cache_sha256')))
            calls.append(dict(case_id=case,request_id=response['request_id'],context_version=g['context_version'],checks=checks,
                              output_stop_reason=g['stop_reason'],selected_count=len(selected)))
    report=dict(scope='Offline exact source and recorded token/cache-identity consistency; not independent tensor recomputation or a new GPU run',
        preserved_original_file_count=len(same),modified_original_files=changed,unexpected_original_changes=unexpected,
        default_hidden_function_signature=defaults,default_hidden_path_ast_equivalent=default_body_same,rest_of_core_ast_unchanged=rest_same,
        actual_fused_calls=len(calls),calls=calls,all_recorded_provenance_checks_pass=all(all(c['checks'].values())for c in calls),
        passed=not unexpected and default_body_same and rest_same and defaults=={'role_system':'WORKER_ROLE_SYSTEM','max_new_tokens':'320'}and len(calls)==12 and all(all(c['checks'].values())for c in calls),
        event_file_sha256=sha(events_path),limitation='Cached tensors are no longer accessible after server expiration. Raw owner hashes are checked for consistency, not independently regenerated.')
    (out/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k]for k in ['passed','preserved_original_file_count','modified_original_files','unexpected_original_changes','default_hidden_path_ast_equivalent','actual_fused_calls','all_recorded_provenance_checks_pass']}))


if __name__=='__main__':main()
