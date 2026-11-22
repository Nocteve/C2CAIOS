"""Six offline next-action calls on fixed real histories, never online repairs."""
import hashlib
import json
from pathlib import Path
from time import perf_counter
from .project import ROOT
from .records import fingerprint


def select_snapshots(events, ids):
    selected=[]
    for cid in ids:
        stream=[e for e in events if e.get('case_id')==cid]
        requests=[e['request'] for e in stream if e['type']=='model_request' and e['request']['kind']=='planner']
        responses={e['response']['request_id']:e['response'] for e in stream if e['type']=='model_response'}
        choices=[]
        for request in requests:
            message=request['messages'][-1]
            observation=json.loads(message['content'])
            observations=observation.get('new_observations',[])
            if any('artifact' in o for o in observations):
                choices.append(('first_consumed_worker_artifact',request))
        if not choices:
            for request in requests:
                observation=json.loads(request['messages'][-1]['content'])
                if any('passed' in o.get('action_result',{}) for o in observation.get('new_observations',[])):
                    choices.append(('first_observed_public_check',request))
        if not choices:
            raise ValueError('Fixed parent lacks the specified structural decision snapshot: '+cid)
        reason,request=choices[0]
        response=responses[request['id']]['result']
        selected.append({'case_id':cid,'selection':reason,'request_id':request['id'],
                         'request':request,'reference':response})
    return selected


def run_planner_call_diagnostic(output_dir):
    import torch
    from .loading import load_models
    from .transcript import aligned_ids
    protocol=json.loads((ROOT/'configs/planner-call-diagnostic.json').read_text())
    record_path=ROOT/protocol['events_path']
    if hashlib.sha256(record_path.read_bytes()).hexdigest()!=protocol['events_sha256']:
        raise ValueError('Diagnostic real-history input changed')
    events=[json.loads(x) for x in record_path.read_text().splitlines()]
    manifest=json.loads((ROOT/'data/job-config-dev-eight/manifest.json').read_text())
    ids=manifest['expected_case_ids'][:2]
    if ids!=protocol['case_ids']:
        raise ValueError('Fixed manifest-prefix selection changed')
    snapshots=select_snapshots(events,ids)
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=False)
    (output/'snapshots.json').write_text(json.dumps(snapshots,ensure_ascii=False,indent=2),encoding='utf-8')
    result={'protocol':protocol,'status':'running','calls':[],'online_history_modified':False,
            'worker_executed':False,'c2c_executed':False,'scope':'offline next-action diagnosis, not benchmark or full system'}
    started=perf_counter()
    config=json.loads((ROOT/'configs/cloud.json').read_text())
    lock=json.loads((ROOT/'configs/cloud.models.lock.json').read_text())
    result['config']=config;result['model_pins']=lock
    result['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [*sorted((ROOT/'src/agent_runtime').glob('*.py')),ROOT/'scripts/run.py']}
    try:
        torch.set_num_threads(4)
        loaded=perf_counter()
        combined,tokenizers=load_models(config,lock)
        model=combined.model_list[1];tokenizer=tokenizers['sender']
        result['loading_seconds']=perf_counter()-loaded
        result['gpu']=torch.cuda.get_device_name()
        for snapshot in snapshots:
            reference=snapshot['reference']['generation']
            messages=snapshot['request']['messages']
            old_text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            old_suffix=aligned_ids(old_text,tokenizer,tokenizers['receiver'])
            original_ids=reference['input_ids']
            if original_ids[-len(old_suffix):]!=old_suffix:
                raise ValueError('Saved actual input does not end in the recorded current invocation')
            historical_ids=original_ids[:-len(old_suffix)]
            for arm in protocol['arms']:
                torch.manual_seed(arm['seed'])
                text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=arm['thinking'])
                ids=historical_ids+aligned_ids(text,tokenizer,tokenizers['receiver'])
                if not arm['thinking'] and ids!=original_ids:
                    raise ValueError('Non-thinking input identity changed')
                if len(ids)+arm['max_new_tokens']>config['max_input_tokens']:
                    raise ValueError('Diagnostic input plus generation exceeds existing context budget')
                kwargs={'do_sample':arm['thinking'],'temperature':.6 if arm['thinking'] else None,
                        'top_p':.95 if arm['thinking'] else None,'top_k':20 if arm['thinking'] else None,
                        'max_new_tokens':arm['max_new_tokens'],'pad_token_id':tokenizer.eos_token_id,
                        'use_cache':True}
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();began=perf_counter()
                with torch.inference_mode():
                    generated=model.generate(input_ids=torch.tensor([ids],device=model.device),
                        attention_mask=torch.ones((1,len(ids)),device=model.device,dtype=torch.long),**kwargs)
                torch.cuda.synchronize();seconds=perf_counter()-began
                emitted=generated[0,len(ids):].tolist();del generated
                raw=tokenizer.decode(emitted,skip_special_tokens=False)
                eos=model.generation_config.eos_token_id
                eos=eos if isinstance(eos,list) else [eos]
                ended=bool(emitted and emitted[-1] in eos)
                close=tokenizer.convert_tokens_to_ids('</think>')
                indices=[i for i,t in enumerate(emitted) if t==close]
                action_ids=emitted[(indices[-1]+1 if indices else 0):]
                final=tokenizer.decode(action_ids,skip_special_tokens=True).strip()
                action,error=None,None
                try:
                    if arm['thinking'] and not indices:raise ValueError('Thinking did not close')
                    if not ended:raise ValueError('Generation truncated')
                    action=json.loads(final)
                    if not isinstance(action,dict):raise ValueError('Expected one JSON action object')
                except (ValueError,TypeError) as exc:
                    error=str(exc)
                row={'case_id':snapshot['case_id'],'snapshot_request':snapshot['request_id'],'arm':arm,
                     'input_ids':ids,'input_sha256':fingerprint(ids),'original_input_sha256':fingerprint(original_ids),
                     'historical_prefix_sha256':fingerprint(historical_ids),'history_tokens_unchanged':True,
                     'prefill_origin':'fresh native HF prefill of exact recorded tokens; NOT original autoregressive Planner KV',
                     'generated_ids':emitted,'raw_output':raw,'final_output':final,'parsed_action':action,
                     'parse_error':error,'stop_reason':'eos' if ended else 'token_limit','output_tokens':len(emitted),
                     'thinking_tokens':indices[-1]+1 if indices else 0,'service_seconds':seconds,
                     'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'output_grammar':False,
                     'action_executed':False,'reference_action':snapshot['reference'].get('action')}
                result['calls'].append(row)
                (output/f"{snapshot['case_id']}-{arm['name']}.json").write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
                print(row['case_id'],arm['name'],row['stop_reason'],row['output_tokens'],round(seconds,3),flush=True)
        result['status']='completed'
    except Exception as exc:
        result.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)})
        raise
    finally:
        result['batch_wall_seconds']=perf_counter()-started
        (output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')


def run_planner_parity(output_dir):
    """Replay original decisions, then compare one structurally selected call per parent."""
    import torch
    from .loading import load_models
    from .transcript import capture_plan,aligned_ids
    from .cache_ops import clone_cache,cache_digest
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import (
        build_token_enforcer_tokenizer_data,build_transformers_prefix_allowed_tokens_fn)
    protocol=json.loads((ROOT/'configs/planner-parity.json').read_text())
    path=ROOT/protocol['events_path']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=protocol['events_sha256']:
        raise ValueError('Frozen source events changed')
    events=[json.loads(line) for line in path.read_text().splitlines()]
    manifest=json.loads((ROOT/'data/job-config-dev-eight/manifest.json').read_text())
    if protocol['case_ids']!=manifest['expected_case_ids']:
        raise ValueError('All eight frozen parents must remain in this diagnosis')
    snapshots=select_snapshots(events,protocol['case_ids'])
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=False)
    (output/'snapshots.json').write_text(json.dumps(snapshots,ensure_ascii=False,indent=2),encoding='utf-8')
    result={'status':'running','protocol':protocol,'cases':[],'expected_parents':len(snapshots),'online_actions_executed':False,
            'cache_scope':'fresh actual autoregressive replay; numerical full-prefill branches are diagnostic only'}
    config=json.loads((ROOT/'configs/cloud.json').read_text())
    lock=json.loads((ROOT/'configs/cloud.models.lock.json').read_text())
    result.update(config=config,model_pins=lock,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [*sorted((ROOT/'src/agent_runtime').glob('*.py')),ROOT/'scripts/run.py']})
    started=perf_counter()
    try:
        torch.set_num_threads(4);torch.manual_seed(config['seed'])
        loaded=perf_counter();combined,tokenizers=load_models(config,lock)
        result['loading_seconds']=perf_counter()-loaded
        model=combined.model_list[1];tokenizer=tokenizers['sender']
        vocabulary=build_token_enforcer_tokenizer_data(tokenizer)
        def allowed(schema):
            return build_transformers_prefix_allowed_tokens_fn(vocabulary,JsonSchemaParser(schema))
        for snapshot in snapshots:
            cid=snapshot['case_id'];previous=None;t=None
            stream=[e for e in events if e.get('case_id')==cid]
            requests=[e['request'] for e in stream if e['type']=='model_request' and e['request']['kind']=='planner']
            references={e['response']['request_id']:e['response']['result'] for e in stream if e['type']=='model_response' and e['response']['kind']=='planner'}
            row={'case_id':cid,'target_request':snapshot['request_id'],'replay':[],'status':'running'}
            result['cases'].append(row)
            for request in requests:
                target=request['id']==snapshot['request_id']
                ref=references[request['id']]['generation']
                ids=ref['input_ids']
                rendered=tokenizer.apply_chat_template(request['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                suffix=aligned_ids(rendered,tokenizer,tokenizers['receiver'])
                reconstructed=(previous.prefix_ids if previous else [])+suffix
                if reconstructed!=ids:
                    row.update(status='replay_input_diverged',request_id=request['id']);break
                prior_hash=cache_digest(previous.source_cache) if target and previous else None
                if target:
                    torch.cuda.synchronize();numeric_at=perf_counter()
                    with torch.inference_mode():
                        kwargs={}
                        if previous:
                            kwargs['past_key_values']=clone_cache(previous.source_cache)
                        incremental=model(input_ids=torch.tensor([suffix],device=model.device),
                            attention_mask=torch.ones((1,len(ids)),device=model.device,dtype=torch.long),
                            use_cache=True,logits_to_keep=1,**kwargs)
                        inc=incremental.logits[:,-1,:].float().clone();del incremental,kwargs
                        native=model(input_ids=torch.tensor([ids],device=model.device),
                            attention_mask=torch.ones((1,len(ids)),device=model.device,dtype=torch.long),
                            use_cache=True,logits_to_keep=1)
                        full=native.logits[:,-1,:].float().clone();del native
                        choices=allowed(request['schema'])(0,torch.tensor(ids,device=model.device))
                        delta=(inc-full).abs()
                        row['numerical']={'max_abs_logit_difference':delta.max().item(),
                            'rms_logit_difference':delta.square().mean().sqrt().item(),
                            'raw_argmax_equal':inc.argmax(-1).item()==full.argmax(-1).item(),
                            'constrained_argmax_equal':inc[:,choices].argmax(-1).item()==full[:,choices].argmax(-1).item(),
                            'incremental_top10':torch.topk(inc,10).indices.tolist(),
                            'full_prefill_top10':torch.topk(full,10).indices.tolist(),
                            'allowed_token_ids':choices,'input_sha256':fingerprint(ids)}
                        torch.save({'incremental':inc.cpu(),'full_prefill':full.cpu()},output/f'{cid}-next-logits.pt')
                        del inc,full,delta
                    torch.cuda.synchronize();row['numerical']['wall_seconds']=perf_counter()-numeric_at
                t=capture_plan(model,tokenizer,tokenizers['receiver'],request['messages'],
                    protocol['max_new_tokens'],config['max_input_tokens'],
                    prior_prefix_ids=previous.prefix_ids if previous else None,
                    prior_cache=previous.source_cache if previous else None,allowed_tokens=allowed(request['schema']))
                same=t.generated_ids==ref['generated_token_ids'] and t.prefix_ids==ref['prefix_ids']
                row['replay'].append({'request_id':request['id'],'input_ids':t.prompt_ids,
                    'generated_ids':t.generated_ids,'output':t.output,'metrics':t.metrics,'matches_original':same})
                if not same:
                    row.update(status='replay_generation_diverged',request_id=request['id']);break
                if target:
                    torch.cuda.synchronize();native_at=perf_counter()
                    with torch.inference_mode():
                        native=model.generate(input_ids=torch.tensor([ids],device=model.device),
                            attention_mask=torch.ones((1,len(ids)),device=model.device,dtype=torch.long),
                            max_new_tokens=protocol['max_new_tokens'],do_sample=False,temperature=None,top_p=None,top_k=None,
                            repetition_penalty=1.0,no_repeat_ngram_size=0,
                            prefix_allowed_tokens_fn=allowed(request['schema']),pad_token_id=tokenizer.eos_token_id,use_cache=True)
                    torch.cuda.synchronize();elapsed=perf_counter()-native_at
                    generated=native[0,len(ids):].tolist();del native
                    row['native_generation']={'generated_ids':generated,'output':tokenizer.decode(generated,skip_special_tokens=True),
                        'matches_incremental':generated==t.generated_ids,'service_seconds':elapsed}
                    row['published_prior_unchanged']=not previous or cache_digest(previous.source_cache)==prior_hash
                    row['status']='compared'
                    break
                previous=t
            (output/f'{cid}.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
            print(cid,row['status'],'native_equal',row.get('native_generation',{}).get('matches_incremental'),flush=True)
            del t,previous
        result['status']='completed'
    except Exception as exc:
        result.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)});raise
    finally:
        result['batch_wall_seconds']=perf_counter()-started
        (output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
