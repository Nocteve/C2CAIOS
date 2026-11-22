"""Run actual Worker processes on all assignments from a fixed recorded dispatch."""
from copy import deepcopy
import hashlib,json,multiprocessing as mp,os,sys
from pathlib import Path
from time import perf_counter
from .project import ROOT
from .records import fingerprint


def run_worker_recovery(output_dir, protocol_path=None):
    from .model_service import AutonomousModels
    from .autonomy import delegated_worker
    from .cache_ops import prefill,cache_digest,cache_bytes
    from .transcript import PlannerTranscript
    from .job_workspace import changed_files,proposal_effect
    protocol_path=ROOT/'configs/worker-recovery.json' if protocol_path is None else Path(protocol_path).resolve()
    if not protocol_path.is_relative_to(ROOT.resolve()):raise ValueError('Protocol must be inside project')
    policy=json.loads(protocol_path.read_text())
    arms=policy.get('arms',[{'name':m,'worker_interface':m,'communication':'text_full'} for m in policy.get('methods',[])])
    if not arms or len({a['name'] for a in arms})!=len(arms):raise ValueError('Unique nonempty arms required')
    for arm in arms:
        if arm['communication'] not in {'text_full','text_brief','c2c_shared'} or arm['worker_interface'] not in {'legacy_json','native_tools'}:
            raise ValueError('Unknown recovery arm')
        if arm.get('worker_context_mode', 'planner_transcript') not in {'planner_transcript', 'independent_session'}:
            raise ValueError('Unknown Worker context mode')
        if arm.get('worker_context_mode') == 'independent_session' and (arm['communication'] == 'c2c_shared' or arm['worker_interface'] != 'native_tools'):
            raise ValueError('Independent session requires native Worker tools and a text communication arm')
    history=ROOT/policy['events_path']
    if hashlib.sha256(history.read_bytes()).hexdigest()!=policy['events_sha256']:
        raise ValueError('Historical dispatch evidence changed')
    events=[json.loads(s) for s in history.read_text().splitlines()]
    original_metadata=next(e for e in events if e['type']=='metadata')
    declarations=[e for e in events if e['type']=='task_declared']
    assert len(declarations)==2, 'Frozen denominator changed'
    config=json.loads((ROOT/'configs/cloud.json').read_text());lock=json.loads((ROOT/'configs/cloud.models.lock.json').read_text())
    assert lock==original_metadata['models'] and config==original_metadata['config']
    public=ROOT/'data/job-config-dev-eight/public.json'
    assert hashlib.sha256(public.read_bytes()).hexdigest()==original_metadata['manifest']['public_sha256']
    cases={c['id']:c for c in json.loads(public.read_text())}
    limits=deepcopy(original_metadata['limits'])
    limits.update(policy.get('limits_overrides',{}))
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=False)
    result={'policy':policy,'models':lock,'config':config,'limits':limits,'scheduler_pid':os.getpid(),
        'expected_assignments':len(declarations),'expected_runs':len(declarations)*len(arms),
        'rows':[],'training':False,'c2c_executed':False,'new_planner_decisions':0,'planner_accepted_proposals':0,
        'source_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*sorted((ROOT/'src/agent_runtime').glob('*.py')),ROOT/'scripts/run.py',protocol_path]}}
    began=perf_counter();host=None
    try:
        start=perf_counter();host=AutonomousModels(config,lock,limits)
        result['loading_seconds']=perf_counter()-start
        import torch
        result['environment']={'gpu':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability(),
            'torch':str(torch.__version__),'python':sys.version,'tf32':torch.backends.cuda.matmul.allow_tf32}
        context=mp.get_context('spawn')
        for declaration in declarations:
            task=deepcopy(declaration['task']);cid=declaration['case_id'];version=task['context_version']
            registered=[{'task_id':e['task']['id'],'context_version':version,'depends_on':e['task']['depends_on']}
                for e in declarations if e['case_id']==cid and e['task']['context_version']==version]
            task['dispatch_receipt']={'function':'delegate','registered_tasks':registered,
                'scope':'Runtime task registration only; no Worker result or task success yet'}
            replies=[e['response']['result'] for e in events if e['type']=='model_response' and e.get('case_id')==cid
                and e['response']['kind']=='planner' and (e['response'].get('result') or {}).get('published_version',{} ) is not None
                and ((e['response'].get('result') or {}).get('published_version') or {}).get('id')==version]
            assert len(replies)==1
            old=replies[0]['generation'];published=replies[0]['published_version']
            assert fingerprint(old['prefix_ids'])==published['prefix_sha256']
            response_event=next(e for e in events if e['type']=='model_response' and e.get('case_id')==cid
                and e['response'].get('result') is replies[0])
            request=next(e['request'] for e in events if e['type']=='model_request' and e.get('case_id')==cid
                and e['request']['id']==response_event['response']['request_id'])
            from .planner_tools import native_tool_input
            messages,tool_signatures=native_tool_input(request['messages'],request['schema'],
                feedback=limits.get('planner_feedback_mode')=='native_tool_response')
            reconstructed=host.tokenizers['sender'].apply_chat_template(messages,tools=tool_signatures,tokenize=True,add_generation_prompt=True,enable_thinking=False)
            assert reconstructed==old['input_ids'], 'Original Planner message/token identity mismatch'
            for arm in arms:
                method=arm['name']
                host.reset();host.limits=deepcopy(limits)
                host.method=arm['communication']
                if arm['worker_interface']=='native_tools':host.limits['worker_interface']='native_tools'
                else:host.limits.pop('worker_interface',None)
                host.limits['worker_context_mode'] = arm.get('worker_context_mode', 'planner_transcript')
                directory=output/(cid+'-'+task['id'])/method;directory.mkdir(parents=True,exist_ok=False)
                row={'case_id':cid,'task_id':task['id'],'method':method,'arm':arm,'task':task,'status':'running',
                    'events':[],'calls':[],'prefix_sha256':fingerprint(old['prefix_ids']),
                    'prefix_origin':'Exact historical actual Planner token IDs re-prefilled; no text re-encoding, no new Planner decision'}
                result['rows'].append(row);started=perf_counter();process=None;parent=None;child=None
                try:
                    source,seconds=prefill(host.sender,old['prefix_ids'])
                    row['historical_prefix_source_prefill_seconds']=seconds
                    t=PlannerTranscript(old['input_ids'],old['generated_token_ids'],old['prefix_ids'],source,old['output'],
                        {'source_cache_origin':row['prefix_origin']},prompt_text=old['prompt_text'])
                    source_audit_started=perf_counter()
                    source_sha=cache_digest(source)
                    row['historical_prefix_source_integrity_audit_seconds']=perf_counter()-source_audit_started
                    metadata={'id':version,'source_cache_sha256':source_sha,'source_cache_bytes':cache_bytes(source),
                        'origin':row['prefix_origin'],'fused':False}
                    host.versions[version]={'transcript':t,'messages':messages,'source_tools':tool_signatures,'metadata':metadata,'native':None,'native_sha256':None}
                    parent,child=context.Pipe()
                    process=context.Process(target=delegated_worker,args=(child,deepcopy(task),cases[cid],host.limits,str(directory/'worker')))
                    process.start();child.close();row['worker_pid']=process.pid
                    while True:
                        if not parent.poll(limits['worker_deadline_seconds']):raise TimeoutError('Worker recovery deadline')
                        event=parent.recv();row['events'].append(event)
                        if event['type']=='model_request':
                            at=perf_counter();used_before=host.used_tokens
                            preparation_before=host.versions[version]['metadata'].get('preparation')
                            try:
                                generated=host.worker(version,event['command'])
                            except Exception as exc:
                                preparation_after=host.versions[version]['metadata'].get('preparation')
                                failure={'error':f'{type(exc).__name__}: {exc}',
                                    'wall_seconds':perf_counter()-at,
                                    'generated_tokens_this_request':host.used_tokens-used_before}
                                if preparation_after is not preparation_before:
                                    failure['failed_context_preparation']=preparation_after
                                row['calls'].append(failure)
                                parent.send({'error':failure['error']})
                            else:
                                result['c2c_executed'] |= bool(generated['generation'].get('published_cache_fused'))
                                row['calls'].append({'result':generated,'wall_seconds':perf_counter()-at,
                                    'generated_tokens_this_request':host.used_tokens-used_before})
                                parent.send(generated['generation'])
                        elif event['type']=='failed':row.update(status='failed',error=event['error']);break
                        elif event['type']=='result':
                            artifact=event['artifact'];row['artifact']=artifact
                            assert fingerprint({k:v for k,v in artifact.items() if k!='sha256'})==artifact['sha256']
                            if artifact['value']['action']=='report':
                                candidate=changed_files(task['snapshot']['files'],artifact['value']['changes'],task['files'])
                                row['proposal_effect']=proposal_effect(cases[cid],task['snapshot']['files'],candidate)
                                (directory/'actual-candidate.json').write_text(json.dumps(candidate,indent=2))
                            row['status']='delivered';break
                except Exception as exc:row.update(status='failed',error=f'{type(exc).__name__}: {exc}')
                finally:
                    if process is not None and process.pid is not None:
                        process.join(timeout=2)
                        if process.is_alive():process.terminate();process.join(timeout=2)
                        if process.is_alive():process.kill();process.join(timeout=2)
                        row.update(worker_reaped=not process.is_alive(),worker_exitcode=process.exitcode)
                    if parent is not None:parent.close()
                    row['cache_audit']=host.end()
                    row['wall_seconds']=perf_counter()-started
                    (directory/'result.json').write_text(json.dumps(row,indent=2))
                print(cid,task['id'],method,row['status'],'candidate_pass',row.get('proposal_effect',{}).get('candidate_parent_check',{}).get('passed'),flush=True)
        result['status']='completed'
    except Exception as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        result['batch_seconds']=perf_counter()-began
        (output/'results.json').write_text(json.dumps(result,indent=2))
