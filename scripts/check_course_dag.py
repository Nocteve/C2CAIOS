"""Concrete dynamic artifact DAG and failed-process recovery component check.

Fixed diagnostic orchestration, NOT a Planner-created application or autonomy
score. All dependency artifacts are real files from actual child processes.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from agent_runtime.course_dag import dependency_status,has_agent_cycle


def child(output,dependencies,fault):
    if fault:os._exit(7)
    public=ROOT/'data/c2c-core-os/public.json';raw=public.read_bytes();case=json.loads(raw)
    received=[]
    for dep in dependencies:
        data=Path(dep).read_bytes();report=json.loads(data)
        if report['source_sha256']!=hashlib.sha256(raw).hexdigest():raise ValueError('dependency belongs to another source')
        received.append(dict(path=str(dep),sha256=hashlib.sha256(data).hexdigest(),producer_pid=report['producer_pid']))
    report=dict(producer_pid=os.getpid(),source_sha256=hashlib.sha256(raw).hexdigest(),
                public_job_ids=[j['id']for j in case['public_background']['tasks']],received_artifacts=received)
    Path(output).write_text(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    case=json.loads((ROOT/'data/c2c-core-os/public.json').read_text());jobs={j['id']:j for j in case['public_background']['tasks']}
    agents={};events=[];ctx=mp.get_context('spawn');start=perf_counter()
    def add(tid,deps):
        agents[tid]=dict(id=tid,task={'depends_on':deps},state='PENDING')
        events.append(dict(event='dynamic_node_added',task_id=tid,depends_on=deps,elapsed=perf_counter()-start))
    def run(tid,fault=False):
        a=agents[tid];blocked,invalid=dependency_status(a['task'],agents,jobs)
        if blocked or invalid or has_agent_cycle(tid,agents):raise ValueError('diagnostic attempted illegal admission')
        dependencies=[out/(d[6:]+'.json')for d in a['task']['depends_on']if d.startswith('agent:')]
        a['state']='RUNNING_TOOL';proc=ctx.Process(target=child,args=(str(out/(tid+'.json')),dependencies,fault));proc.start();proc.join(20)
        if proc.is_alive():proc.kill();proc.join(5)
        a.update(state='REPORTED' if proc.exitcode==0 else 'FAILED',pid=proc.pid,exitcode=proc.exitcode)
        events.append(dict(event='actual_process_terminal',task_id=tid,pid=proc.pid,exitcode=proc.exitcode,elapsed=perf_counter()-start))
    add('scan',[]);run('scan')
    add('hash',['agent:scan']);add('verify',['agent:hash'])
    before=dependency_status(agents['verify']['task'],agents,jobs)
    run('hash',fault=True)
    failed_dependency=dependency_status(agents['verify']['task'],agents,jobs)
    # Explicit diagnostic controller repair; no claim that a model decided it.
    add('hash_retry',['agent:scan']);run('hash_retry')
    agents['verify']['state']='CANCELLED'
    add('verify_repair',['agent:hash_retry']);run('verify_repair')
    blocked_public=dependency_status({'depends_on':['job:encode']},agents,jobs)
    add('cycle_x',['agent:cycle_y']);add('cycle_y',['agent:cycle_x'])
    cycle=has_agent_cycle('cycle_x',agents)
    result=dict(scope='Deterministic OS mechanism check; no Planner/model calls; not an autonomy success',
        passed=bool(before[0] and failed_dependency[0] and blocked_public[0] and cycle and agents['verify_repair']['state']=='REPORTED'),
        actual_agents=list(agents.values()),events=events,blocked_before_delivery=before,failed_dependency_preserved=failed_dependency,
        public_running_job_not_marked_success=blocked_public,cycle_rejected=cycle,
        final_actual_artifact=json.loads((out/'verify_repair.json').read_text()))
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(passed=result['passed'],pids=[a.get('pid')for a in agents.values()if a.get('pid')])))


if __name__=='__main__':main()
