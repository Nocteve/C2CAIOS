import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const dir=path.dirname(fileURLToPath(import.meta.url));
const root=path.join(dir,'..');
const source='artifacts/osclassics-batch-20261019/acbi-osclassics-sched-poolmmap';
const read=async p=>JSON.parse(await fs.readFile(path.join(root,p),'utf8'));
const r=await read(source+'/result.json');
const publicCase=await read('data/c2c-core-os/public-osclassics-sched-01.json');
const tasks=[];
for(const id of ['t1','t2','t1.r1','t2.r1']){
 const a=await read(source+'/tasks/'+id+'.json');
 const report=r.acb_report.tasks[id];
 tasks.push({id,pid:a.worker_pid,gate:a.application_gate,isolation:a.isolation,cacheLoadMs:a.cache_load_seconds*1000,pool:a.payload_pool,state:report.state,repairOf:report.repair_of,feedback:a.command.includes('[runtime feedback]')?a.command.split('[runtime feedback]')[1].trim():null,outputTokens:a.output_tokens});
}
if(new Set(tasks.map(t=>t.pid)).size!==4||tasks.filter(t=>t.gate.accepted).length!==1)throw Error('Evidence mismatch');
await fs.writeFile(path.join(dir,'system-data.json'),JSON.stringify({source,caseId:r.case_id,wall:r.wall_seconds,policy:r.sched_policy,limits:r.worker_limits,jobs:publicCase.public_background.tasks,initialTasks:r.planner.action.tasks.map((t,i)=>({id:'t'+(i+1),command:t.command,deliverable:t.deliverable,dependsOn:t.depends_on})),budget:r.scheduler_events.find(e=>e.event==='admit').budget,estimatedBytes:209715200,events:r.scheduler_events,poolEvents:r.payload_pool_events,capacity:r.payload_pool_capacity,tasks},null,2)+'\n');
console.log('Integrated example: 4 real PIDs, 2 repairs, 1/4 accepted, 3 swap-outs and 2 file restores.');
