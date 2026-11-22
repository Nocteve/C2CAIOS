"""Actual staged hashing-process comparison: same READY boundaries and inputs.

Mechanism microbenchmark, not a model application or autonomy demonstration.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
from time import perf_counter,process_time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from agent_runtime.course_scheduler import pick_ready


def tool(pipe,source):
    raw=Path(source).read_bytes()
    while True:
        request=pipe.recv()
        if request is None:
            return
        before=process_time()
        for _ in range(request['iterations']):hashlib.sha256(raw).digest()
        pipe.send(dict(pid=os.getpid(),stage=request['stage'],source_sha256=hashlib.sha256(raw).hexdigest(),
                       actual_cpu_seconds=process_time()-before))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'data/c2c-core-os/public.json';expected=hashlib.sha256(source.read_bytes()).hexdigest()
    workload=[('long',4,100000),('short',1,12000),('medium',3,40000)]
    records=[]
    ctx=mp.get_context('spawn')
    for policy in ['fifo','vservice']:
        acbs=[];owners={};events=[];sequence=0;started=perf_counter()
        try:
            for jid,stages,iterations in workload:
                parent,child=ctx.Pipe();proc=ctx.Process(target=tool,args=(child,str(source)));proc.start();child.close()
                sequence+=1
                acbs.append(dict(id=jid,state='READY',ready_sequence=sequence,virtual_service=0.0,
                    weight=1,stages=stages,iterations=iterations,completed_stages=0,pid=proc.pid,ready_since=perf_counter(),wait_seconds=0.0))
                owners[jid]=(proc,parent)
            while (a:=pick_ready(acbs,policy)) is not None:
                a['wait_seconds']+=perf_counter()-a['ready_since'];a['state']='RUNNING_TOOL'
                stage=a['completed_stages']+1;proc,pipe=owners[a['id']];wall=perf_counter()
                pipe.send(dict(stage=stage,iterations=a['iterations']))
                if not pipe.poll(30):raise TimeoutError('actual hashing process stage deadline')
                result=pipe.recv()
                if result['source_sha256']!=expected:raise ValueError('wrong actual file digest')
                a['virtual_service']+=result['actual_cpu_seconds'];a['completed_stages']=stage
                events.append(dict(task_id=a['id'],elapsed=perf_counter()-started,stage_wall_seconds=perf_counter()-wall,
                    service=result,virtual_service_after=a['virtual_service'],selection_policy=policy))
                if stage==a['stages']:
                    a.update(state='SUCCEEDED',completed_elapsed=perf_counter()-started)
                else:
                    sequence+=1;a.update(state='READY',ready_sequence=sequence,ready_since=perf_counter())
            for proc,pipe in owners.values():pipe.send(None);proc.join(5)
            row=dict(policy=policy,acbs=acbs,events=events,wall_seconds=perf_counter()-started,
                     stage_order=[x['task_id'] for x in events],all_real_file_digests_match=True)
            records.append(row)
            (out/(policy+'.json')).write_text(json.dumps(row,indent=2))
        finally:
            for proc,pipe in owners.values():
                if proc.is_alive():proc.kill();proc.join(5)
                pipe.close()
    result=dict(scope='CPU tool-stage scheduling only; no model calls, no autonomous Planner claims',
        workload=workload,public_source_sha256=expected,policies=records,
        comparison='Both policies reselect at identical real stage boundaries; FIFO requeues at tail; vservice counts actual process CPU time',
        limitations='One paired engineering observation; no statistical or model-performance claim')
    (out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps([dict(policy=r['policy'],order=r['stage_order'])for r in records]))


if __name__=='__main__':main()
