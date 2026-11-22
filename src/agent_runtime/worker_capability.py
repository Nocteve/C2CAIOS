"""One preregistered Worker-component baseline, optional small training, and paired validation."""
from copy import deepcopy
import hashlib,json,multiprocessing as mp,os
from pathlib import Path
from time import perf_counter
from .project import ROOT
from .records import fingerprint


def run_worker_capability(output_dir,protocol_path):
    from .autonomy import delegated_worker
    from .model_service import gpu_service
    from .ledger_contract import ledger_outcome
    protocol_path=Path(protocol_path).resolve()
    if not protocol_path.is_relative_to(ROOT.resolve()):raise ValueError('Project-local frozen protocol required')
    policy=json.loads(protocol_path.read_text())
    for relative,sha in policy['input_sha256'].items():
        path=(ROOT/relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
            raise ValueError('Frozen component input changed')
    public=json.loads((ROOT/policy['public_path']).read_text())
    cases={case['id']:case for case in public}
    expected=policy['calibration_case_ids']+policy['validation_case_ids']
    if len(set(cases))!=len(public) or set(expected)!=set(cases):raise ValueError('Frozen public denominator changed')
    train_families=set(policy['training']['train_families'])
    if any(cases[cid]['family'] in train_families for cid in policy['validation_case_ids']):
        raise ValueError('Training/validation family overlap')
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=False)
    config=json.loads((ROOT/'configs/cloud.json').read_text());lock=json.loads((ROOT/'configs/cloud.models.lock.json').read_text())
    limits=policy['limits'];context=mp.get_context('spawn');parent,child=context.Pipe();canceled=context.Array('b',2)
    service=context.Process(target=gpu_service,args=(child,config,lock,limits,canceled),name='GPU-owner')
    sources=[*sorted((ROOT/'src/agent_runtime').glob('*.py')),ROOT/'scripts/run.py',protocol_path]
    result={'kind':'worker_component_development','status':'running','protocol':policy,'models':lock,'config':config,
        'source_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'scheduler_pid':os.getpid(),
        'calibration':[{'id':cid,'status':'not_run','semantic_passed':False} for cid in policy['calibration_case_ids']],
        'validation':{arm:[{'id':cid,'status':'not_run','semantic_passed':False} for cid in policy['validation_case_ids']] for arm in ('base','adapted')},
        'autonomous_delegation':False,'c2c':False,'training':None,'formal_test':False,
        'expected_validation_rows_per_arm':len(policy['validation_case_ids'])}
    began=perf_counter();sequence=0;logging_seconds=0;service_failed=False;log=(output/'events.jsonl').open('x')
    def emit(event_type,**fields):
        nonlocal logging_seconds
        at=perf_counter();log.write(json.dumps({'type':event_type,**fields})+'\n');log.flush();logging_seconds+=perf_counter()-at
    def request(kind,**fields):
        nonlocal sequence,service_failed
        sequence+=1;packet={'id':f'req{sequence}','kind':kind,'submitted_at':perf_counter(),**fields}
        emit('model_request',request=packet);parent.send(packet)
        deadline=limits['training_deadline_seconds'] if kind=='component_train' else limits['model_service_deadline_seconds']
        if not parent.poll(deadline):
            service_failed=True
            raise TimeoutError('Actual GPU service deadline')
        response=parent.recv();emit('model_response',response=response)
        if response.get('fatal'):
            service_failed=True
            raise RuntimeError(str(response.get('error')))
        return response
    def run_case(cid,arm,row):
        case=cases[cid];directory=output/arm/cid;directory.mkdir(parents=True,exist_ok=False)
        task={'id':cid,'command':case['goal'],'files':list(case['files']),'depends_on':[],
            'deliverable':'Actual local ledger finding and explicit proposed edits, or the concrete missing public fact.',
            'success_criteria':'Satisfy every public receipt expression while preserving immutable fields. Do not invent missing facts.',
            'snapshot':{'version':0,'files':deepcopy(case['files']),'sha256':fingerprint(case['files'])},
            'context_version':'public-component-'+cid,'dependency_artifacts':[]}
        row.update(id=cid,family=case['family'],arm=arm,case_sha256=fingerprint(case),task=task,
            status='running',calls=[],events=[],semantic_passed=False)
        started=perf_counter();process=None;connection=None
        try:
            response=request('begin',method='text_full')
            if response.get('error'):raise ValueError(str(response['error']))
            connection,worker=context.Pipe()
            process=context.Process(target=delegated_worker,args=(worker,task,case,limits,str(directory/'worker')),name='Worker-'+cid)
            process.start();worker.close();row['worker_pid']=process.pid
            while True:
                if not connection.poll(limits['worker_deadline_seconds']):raise TimeoutError('Actual Worker lifecycle deadline')
                event=connection.recv();row['events'].append(event);emit('worker_event',case_id=cid,arm=arm,event=event)
                if event['type']=='model_request':
                    response=request('component_worker',case=case,command=event['command'],adapted=arm=='adapted')
                    row['calls'].append(response)
                    generation=(response.get('result') or {}).get('generation')
                    connection.send(generation if generation is not None else {'error':str(response.get('error'))})
                elif event['type']=='failed':row.update(status='failed',error=event['error'],failure_stage=event.get('failure_stage','worker_runtime'));break
                elif event['type']=='result':
                    row.update(status='delivered',artifact=event['artifact']);break
        except Exception as exc:
            row.update(status='failed',error=f'{type(exc).__name__}: {exc}',failure_stage='worker_or_service_runtime')
            if isinstance(exc,RuntimeError):raise
        finally:
            if process is not None and process.pid:
                process.join(timeout=2)
                if process.is_alive():process.terminate();process.join(timeout=2)
                if process.is_alive():process.kill();process.join(timeout=2)
                row.update(worker_reaped=not process.is_alive(),worker_exitcode=process.exitcode)
                process.close()
            if connection is not None:connection.close()
            try:
                if service_failed:raise ValueError('GPU service failed; cache audit unavailable')
                response=request('end');row['cache_audit']=response
                if response.get('error'):raise ValueError(str(response['error']))
            except Exception as exc:
                row['audit_error']=f'{type(exc).__name__}: {exc}'
            try:
                row['outcome']=ledger_outcome(case,task,row.get('artifact'));row['semantic_passed']=row['outcome']['semantic_passed']
            except Exception as exc:
                row.update(status='failed',error=f'{type(exc).__name__}: {exc}',failure_stage='deliverable_integrity')
                row['outcome']={'semantic_passed':False,'artifact_present':bool(row.get('artifact'))}
            if row['outcome'].get('candidate_files'):
                (directory/'actual-candidate.json').write_text(json.dumps(row['outcome']['candidate_files'],indent=2))
            checks=[e for e in row['events'] if e['type']=='tool_result' and e['worker_action']['action']=='check']
            row['failed_check_seen']=any(not e['tool_result']['passed'] for e in checks)
            row['recovered_after_failed_check']=row['failed_check_seen'] and row['semantic_passed']
            row['wall_seconds']=perf_counter()-started
            (directory/'result.json').write_text(json.dumps(row,indent=2))
        print(arm,cid,row['status'],'semantic_passed',row['semantic_passed'],flush=True)
        return row
    try:
        emit('metadata',**{k:v for k,v in result.items() if k not in {'calibration','validation','training'}})
        service.start();child.close()
        if not parent.poll(limits['model_service_deadline_seconds']):raise TimeoutError('GPU owner startup')
        ready=parent.recv();result['service']=ready;emit('model_service_started',event=ready)
        if ready.get('type')!='ready':raise RuntimeError(str(ready))
        for index,cid in enumerate(policy['calibration_case_ids']):run_case(cid,'calibration',result['calibration'][index])
        infrastructure_failure=any(r.get('audit_error') or (r['status']=='failed' and r.get('failure_stage')!='model_output') for r in result['calibration'])
        if infrastructure_failure:
            result['training']={'blocked':'Infrastructure or integrity failure is not a model capability gap'}
            raise ValueError('Calibration infrastructure failure blocks adaptation')
        gap=any(not r['semantic_passed'] for r in result['calibration'])
        result['training_triggered_by_actual_gap']=gap
        if gap:
            response=request('component_train',policy=policy['training'],directory=str(output/'training'),
                training_path=str(ROOT/policy['training_path']),training_sha256=policy['input_sha256'][policy['training_path']])
            result['training']=response
            if response.get('error'):raise ValueError('Fixed training failed: '+str(response['error']))
        else:result['training']={'skipped':'All frozen calibration cases passed; no necessary adaptation'}
        # Validation is first consumed after the final checkpoint is fixed. No selection or follow-up tuning.
        for arm in ('base','adapted') if gap else ('base',):
            for index,cid in enumerate(policy['validation_case_ids']):run_case(cid,arm,result['validation'][arm][index])
        result['status']='completed'
    except Exception as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}');emit('run_failed',error=result['error'])
        raise
    finally:
        if service.pid:
            try:parent.send({'kind':'stop'})
            except (EOFError,BrokenPipeError,OSError):pass
            service.join(timeout=3)
            if service.is_alive():service.terminate();service.join(timeout=3)
            if service.is_alive():service.kill();service.join(timeout=3)
            result.update(model_service_reaped=not service.is_alive(),model_service_exitcode=service.exitcode)
            service.close()
        parent.close();result.update(batch_wall_seconds=perf_counter()-began,event_logging_seconds=logging_seconds)
        (output/'results.json').write_text(json.dumps(result,indent=2));log.close()
