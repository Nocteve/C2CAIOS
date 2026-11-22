"""Fixed real-delegation communication replay; not an autonomous system benchmark."""
import hashlib
import json
import re
from pathlib import Path
from time import perf_counter

from .project import ROOT
from .records import fingerprint
from .delegation_contract import SHARED_SYSTEM, worker_command, parse_worker_transport
from .job_workspace import changed_files, check_files


def value_spans(text):
    """Locate JSON values in their actual spelling, without serializing them again."""
    decoder=json.JSONDecoder();spans={}
    def space(i):
        while i<len(text) and text[i].isspace():i+=1
        return i
    def walk(i,path):
        i=space(i);value,end=decoder.raw_decode(text,i);spans[path]=(i,end)
        if isinstance(value,dict):
            pos=space(i+1)
            while text[pos]!='}':
                key,pos=decoder.raw_decode(text,pos);pos=space(pos)
                if text[pos]!=':':raise ValueError('Missing JSON colon')
                pos=space(walk(pos+1,path+(key,)))
                if text[pos]==',':pos=space(pos+1)
                elif text[pos]!='}':raise ValueError('Invalid object boundary')
        elif isinstance(value,list):
            pos=space(i+1);index=0
            while text[pos]!=']':
                pos=space(walk(pos,path+(index,)));index+=1
                if text[pos]==',':pos=space(pos+1)
                elif text[pos]!=']':raise ValueError('Invalid array boundary')
        return end
    walk(0,())
    return spans


def state_planning_scope(t,messages,tokenizer,*,include_planning=True):
    prompt=t.planning_input_text if t.planning_input_text is not None else t.prompt_text
    prompt_ids=t.planning_input_ids if t.planning_input_ids is not None else t.prompt_ids
    if prompt is None:
        raise ValueError('Cannot identify the real fresh-state content')
    ranges=[];locations=[]
    for key in ('public_background','current_state'):
        matches=[]
        for index,message in enumerate(messages):
            if message['role'] not in {'user','tool'}:continue
            content=message.get('content','')
            try:spans=value_spans(content)
            except (ValueError,IndexError):continue
            path=(key,) if message['role']=='user' else ('current_runtime_observation',key)
            if path not in spans:continue
            if prompt.count(content)!=1:
                raise ValueError('Actual selected message content is missing or ambiguous')
            origin=prompt.index(content);lo,hi=spans[path]
            matches.append((origin+lo,origin+hi))
            locations.append({'field':key,'message_index':index,'role':message['role'],'json_path':list(path)})
        if len(matches)!=1:raise ValueError('Expected exactly one actual '+key+' value')
        ranges.extend(matches)
    encoded=tokenizer(prompt,add_special_tokens=False,return_offsets_mapping=True)
    if encoded['input_ids']!=prompt_ids:raise ValueError('Actual prompt token/offset identity differs')
    if t.prefix_ids[:len(prompt_ids)]!=prompt_ids:raise ValueError('Planning input is not the actual published prefix')
    background={i for i,(a,b) in enumerate(encoded['offset_mapping']) if any(lo<=a<b<=hi for lo,hi in ranges)}
    background_boundary={i for i,(a,b) in enumerate(encoded['offset_mapping'])
        if a<b and any(a<hi and lo<b for lo,hi in ranges)}-background
    generated=tokenizer.decode(t.generated_ids,skip_special_tokens=False)
    action_output=t.action_output if t.action_output is not None else t.output
    match=re.fullmatch(r'\s*<tool_call>\s*(\{[\s\S]*\})\s*</tool_call>\s*',action_output)
    if match:
        body=match.group(1)
        start=0
        if t.action_output is not None:
            closed=t.analysis_token_range[1]-len(t.prompt_ids)+1
            start=len(tokenizer.decode(t.generated_ids[:closed],skip_special_tokens=False))
        if generated[start:].count(body)!=1:raise ValueError('Native dispatch JSON is ambiguous in actual generated tokens')
        origin=generated.index(body,start);call=json.loads(body)
        if set(call)!={'name','arguments'} or call['name']!='delegate':
            raise ValueError('Scope requires an actual delegate call')
        action={'action':call['name'],**call['arguments']}
        spans={path[1:]:(origin+a,origin+b) for path,(a,b) in value_spans(body).items() if path[:1]==('arguments',)}
    else:
        spans=value_spans(generated)
        action=json.loads(t.output)
    plan_ranges=[(a+1,b-1) for path,(a,b) in spans.items()
        if include_planning and (path==('reason',) or
            (len(path)==3 and path[0]=='tasks' and path[2] in {'command','deliverable','success_criteria'}))]
    if include_planning and len(plan_ranges)!=1+3*len(action['tasks']):
        raise ValueError('Missing actual planning string spans')
    prefixes=[tokenizer.decode(t.generated_ids[:i],skip_special_tokens=False) for i in range(len(t.generated_ids)+1)]
    if any(not generated.startswith(p) for p in prefixes):
        raise ValueError('Actual generated-token text has ambiguous partial decoding; no re-encoding fallback')
    lengths=list(map(len,prefixes))
    if lengths!=sorted(lengths):raise ValueError('Nonmonotonic actual token character boundaries')
    planning={len(t.prompt_ids)+i for i,(a,b) in enumerate(zip(lengths,lengths[1:]))
              if any(lo<=a<b<=hi for lo,hi in plan_ranges)}
    planning_boundary={len(t.prompt_ids)+i for i,(a,b) in enumerate(zip(lengths,lengths[1:]))
        if a<b and any(a<hi and lo<b for lo,hi in plan_ranges)}-planning
    if not background or (include_planning and not planning):
        raise ValueError('Empty public-state or actual-planning selection')
    analysis=set()
    if t.analysis_token_range is not None:
        lo,hi=t.analysis_token_range
        prior_analysis=len(prompt_ids)<=lo<hi<len(t.prompt_ids)
        native_analysis=(t.action_output is not None and lo==len(t.prompt_ids)+1 and lo<=hi<len(t.prefix_ids))
        if not (prior_analysis or native_analysis):
            raise ValueError('Invalid actual analysis range')
        analysis=set(range(lo,hi))
    selected=tuple(sorted(background|planning|analysis))
    return selected,{'background_indices':sorted(background),'actual_planning_indices':sorted(planning),
        'input_value_locations':locations,'generated_dispatch_envelope':'native_tool_call' if match else 'action_json',
        'actual_analysis_indices':sorted(analysis),
        'prefix_tokens':len(t.prefix_ids),'selected_tokens':len(selected),
        'boundary_policy':'only tokens fully inside an identified value span are selected; overlapping boundary tokens stay native',
        'native_boundary_overlap_indices':sorted(background_boundary|planning_boundary),
        'policy':('public/current-state values, actual generated dispatch strings, and actual analysis tokens when present; no recoding'
                  if include_planning else
                  'public/current-state values and actual analysis tokens only; Planner dispatch remains short-text control plane; no recoding'),
        'dispatch_semantics_included':include_planning,
        'current_state_also_contains_control_records_and_old_claims':True,
        'prior_control_information_not_removed_from_selected_states':True,
        'learned_scoring':False,'sparse_projection':False,'physical_zero_copy':False}


def run_communication_probe(output_dir):
    import torch
    from .loading import load_models
    from .transcript import capture_plan,worker_ids
    from .cache_ops import prefill,cache_digest,cache_bytes
    from .kv import PlanningPayload,generate_worker,fuse_context
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data,build_transformers_prefix_allowed_tokens_fn
    protocol=json.loads((ROOT/'configs/communication-probe.json').read_text())
    path=ROOT/protocol['events_path']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=protocol['events_sha256']:raise ValueError('Frozen source changed')
    events=[json.loads(line) for line in path.read_text().splitlines()]
    metadata=next(e for e in events if e['type']=='metadata')
    config=json.loads((ROOT/'configs/cloud.json').read_text());lock=json.loads((ROOT/'configs/cloud.models.lock.json').read_text())
    if metadata['config']!=config or metadata['models']!=lock:raise ValueError('Models or inference configuration changed')
    public=ROOT/'data/job-config-dev-eight/public.json'
    if hashlib.sha256(public.read_bytes()).hexdigest()!=metadata['manifest']['public_sha256']:raise ValueError('Public tasks changed')
    cases=json.loads(public.read_text())
    if [c['id'] for c in cases]!=protocol['case_ids']:raise ValueError('Parent denominator changed')
    for name in ('loading','transcript','cache_ops','kv','model_service','delegation_contract'):
        relative=f'src/agent_runtime/{name}.py'
        if metadata['source_sha256'][relative]!=hashlib.sha256((ROOT/relative).read_bytes()).hexdigest():
            raise ValueError('Replay-critical source changed: '+relative)
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=False)
    result={'status':'running','protocol':protocol,'config':config,'model_pins':lock,
        'rows':[{'case_id':c['id'],'status':'not_run','arms':{}} for c in cases],
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*sorted((ROOT/'src/agent_runtime').glob('*.py')),ROOT/'scripts/run.py',ROOT/'configs/communication-probe.json']},
        'scope':'fixed real first-delegation mechanism comparison, no global commits or autonomous replanning',
        'handoff_is_extra_source_task_assistance':True}
    started=perf_counter()
    try:
        torch.set_num_threads(4);torch.manual_seed(config['seed'])
        load_started=perf_counter();combined,tokenizers=load_models(config,lock);combined.multi_source_fusion_mode='sequential'
        result['loading_seconds']=perf_counter()-load_started
        receiver,sender=combined.model_list;tok=tokenizers['sender'];rtok=tokenizers['receiver']
        setup=perf_counter();vocabulary=build_token_enforcer_tokenizer_data(tok)
        result['grammar_setup_seconds']=perf_counter()-setup
        for case,row in zip(cases,result['rows']):
            stream=[e for e in events if e.get('case_id')==case['id']]
            declarations=[e for e in stream if e['type']=='task_declared']
            if not declarations:
                row['status']='no_actual_delegation';continue
            task=declarations[0]['task'];row['task']=task
            if task['depends_on'] or task['dependency_artifacts']:raise ValueError('First task needs unexpected prior artifacts')
            row['arms']={m:{'status':'not_run','calls':[],'tools':[]} for m in protocol['methods']}
            responses=[e['response'] for e in stream if e['type']=='model_response' and e['response']['kind']=='planner']
            response=next(r for r in responses if ((r.get('result') or {}).get('published_version') or {}).get('id')==task['context_version'])
            request=next(e['request'] for e in stream if e['type']=='model_request' and e['request']['id']==response['request_id'])
            reference=response['result'];row['source_request_id']=request['id']
            row_started=perf_counter();payloads={};hashes={};t=None;brief=None;native=None;fused=None
            try:
                allowed=build_transformers_prefix_allowed_tokens_fn(vocabulary,JsonSchemaParser(request['schema']))
                began=perf_counter()
                t=capture_plan(sender,tok,rtok,request['messages'],metadata['limits']['planner_max_new_tokens'],
                               config['max_input_tokens'],allowed_tokens=allowed)
                row['source_replay_seconds']=perf_counter()-began
                row['source_generation']={'input_ids':t.prompt_ids,'generated_ids':t.generated_ids,'prefix_ids':t.prefix_ids,
                    'output':t.output,'metrics':t.metrics,'input_messages':request['messages']}
                original=reference['generation']
                identity=(t.prompt_ids==original['input_ids'] and t.generated_ids==original['generated_token_ids'] and t.prefix_ids==original['prefix_ids'])
                audit_started=perf_counter();source_sha=cache_digest(t.source_cache)
                row['integrity_audit_seconds']=perf_counter()-audit_started
                row['replay_token_identity']=identity
                row['source_cache_sha256']=source_sha
                row['source_cache_matches_original']=source_sha==reference['published_version']['source_cache_sha256']
                if not identity or not row['source_cache_matches_original']:
                    raise ValueError('Original actual source token/cache identity did not reproduce; no forced reference tokens')
                row['source_cache_bytes']=cache_bytes(t.source_cache)
                native,seconds=prefill(receiver,t.prefix_ids);row['receiver_full_background_prefill_seconds']=seconds
                payloads['text_full']=PlanningPayload(tuple(t.prefix_ids),native)
                row['arms']['text_full']['preparation_seconds']=seconds
                audit_started=perf_counter();hashes['text_full']=cache_digest(native)
                row['integrity_audit_seconds']+=perf_counter()-audit_started
                prep_started=perf_counter()
                try:
                    scope_started=perf_counter();indices,scope=state_planning_scope(t,request['messages'],tok)
                    scope_seconds=perf_counter()-scope_started;timings={}
                    fused,fusion_seconds=fuse_context(combined,t.source_cache,native,token_indices=indices,timings=timings)
                    row['scope']=scope;row['scope_seconds']=scope_seconds;row['fusion_timings']=timings
                    payloads['c2c_shared']=PlanningPayload(tuple(t.prefix_ids),fused,True,indices)
                    row['arms']['c2c_shared']['preparation_seconds']=seconds+scope_seconds+fusion_seconds
                    audit_started=perf_counter();hashes['c2c_shared']=cache_digest(fused)
                    row['integrity_audit_seconds']+=perf_counter()-audit_started
                except Exception as exc:
                    payloads.pop('c2c_shared',None);hashes.pop('c2c_shared',None)
                    row['arms']['c2c_shared'].update(status='preparation_failed',error=str(exc))
                finally:
                    row['arms']['c2c_shared']['observed_extra_preparation_wall_seconds']=perf_counter()-prep_started
                prep_started=perf_counter()
                try:
                    handoff_request=json.dumps({'invocation_role':'Planner','current_operation':'prepare_worker_handoff',
                        'instructions':'For this invocation, write a standalone useful handoff in plain text, not a next scheduler action. Include the facts, constraints, evidence and working guidance this exact task needs. Useful calculations or candidate edits are allowed when supported. Distinguish missing information and unverified claims. Do not merely repeat the instruction. The Worker will separately receive its task command.',
                        'task':{k:task[k] for k in ('command','files','deliverable','success_criteria')}},ensure_ascii=False)
                    began=perf_counter()
                    brief=capture_plan(sender,tok,rtok,[{'role':'user','content':handoff_request}],protocol['handoff_max_new_tokens'],
                        config['max_input_tokens'],producer_role='practical handoff',prior_prefix_ids=t.prefix_ids,prior_cache=t.source_cache)
                    row['handoff_seconds']=perf_counter()-began
                    row['handoff']={'request':handoff_request,'output':brief.output,'input_ids':brief.prompt_ids,
                        'generated_ids':brief.generated_ids,'metrics':brief.metrics}
                    if not brief.metrics['complete_assistant_turn'] or not brief.output.strip():raise ValueError('Incomplete actual handoff')
                    messages=[{'role':'system','content':SHARED_SYSTEM},{'role':'user','content':brief.output}]
                    brief_ids=rtok.apply_chat_template(messages,tokenize=True,add_generation_prompt=False)
                    cache,seconds=prefill(receiver,brief_ids)
                    payloads['text_brief']=PlanningPayload(tuple(brief_ids),cache)
                    row['arms']['text_brief']['preparation_seconds']=row['handoff_seconds']+seconds
                    row['handoff_receiver_prefill_seconds']=seconds
                    audit_started=perf_counter();hashes['text_brief']=cache_digest(cache)
                    row['integrity_audit_seconds']+=perf_counter()-audit_started
                    del cache
                except Exception as exc:
                    payloads.pop('text_brief',None);hashes.pop('text_brief',None)
                    row['arms']['text_brief'].update(status='preparation_failed',error=str(exc))
                finally:
                    row['arms']['text_brief']['observed_preparation_wall_seconds']=perf_counter()-prep_started
                for method in protocol['methods']:
                    arm=row['arms'][method]
                    if method not in payloads:continue
                    observations=[];arm_started=perf_counter();arm['cache_bytes']=payloads[method].nbytes
                    try:
                        for call in range(protocol['worker_max_model_calls']):
                            command=worker_command(task,observations)
                            ids=worker_ids(rtok,list(payloads[method].prefix_ids),command,config['max_input_tokens'])
                            if len(ids)+protocol['worker_max_new_tokens']>config['max_input_tokens']:raise ValueError('Worker context budget exceeded')
                            generated=generate_worker(combined,tokenizers,ids,payloads[method],
                                {'text_full':'text_full_cached','text_brief':'text_brief_cached','c2c_shared':'c2c_shared'}[method],protocol['worker_max_new_tokens'])
                            generated.update(input_ids=ids,command=command,output_constraints=False)
                            arm['calls'].append(generated)
                            value,validation=parse_worker_transport(generated,task)
                            generated.update(actual_action=value,output_validation=validation)
                            if value['action'] in {'report','need_info'}:
                                arm.update(status='terminal_artifact',artifact=value);break
                            tool_started=perf_counter()
                            tool_result=({'version':task['snapshot']['version'],'files':{n:task['snapshot']['files'][n] for n in task['files']}}
                                if value['action']=='inspect' else check_files(case,changed_files(task['snapshot']['files'],value['changes'],task['files'])))
                            observation={'worker_action':value,'tool_result':tool_result};observations.append(observation)
                            arm['tools'].append({**observation,'seconds':perf_counter()-tool_started})
                        else:arm.update(status='call_budget_exhausted',error='No actual terminal artifact in two calls')
                    except Exception as exc:arm.update(status='failed',error=str(exc))
                    arm['worker_wall_seconds']=perf_counter()-arm_started
                audit_started=perf_counter()
                row['cache_audit']={'source_unchanged':cache_digest(t.source_cache)==source_sha,
                    'receivers_unchanged':{m:cache_digest(payloads[m].receiver_cache)==sha for m,sha in hashes.items()}}
                row['integrity_audit_seconds']+=perf_counter()-audit_started
                if not row['cache_audit']['source_unchanged'] or not all(row['cache_audit']['receivers_unchanged'].values()):
                    raise ValueError('Published source/receiver cache changed')
                row['status']='compared'
            except Exception as exc:row.update(status='failed',error=str(exc))
            finally:
                row['wall_seconds']=perf_counter()-row_started
                (output/f"{case['id']}.json").write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
                payloads.clear();t=None;brief=None;native=None;fused=None
            print(case['id'],row['status'],{m:a['status'] for m,a in row['arms'].items()},flush=True)
        result['status']='completed'
    except Exception as exc:
        result.update(status='failed',error=str(exc));raise
    finally:
        result['batch_wall_seconds']=perf_counter()-started
        (output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
