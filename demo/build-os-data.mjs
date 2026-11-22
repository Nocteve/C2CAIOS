import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const dir=path.dirname(fileURLToPath(import.meta.url));
const source='artifacts/os-sched-evidence-20261019/checks.json';
const workloadSource='scripts/check_os_sched.py';
const evidence=JSON.parse(await fs.readFile(path.join(dir,'..',source),'utf8'));
const script=await fs.readFile(path.join(dir,'..',workloadSource),'utf8');
const workload=script.match(/service = \{([\s\S]*?)\}/)[1];
const serviceMs=Object.fromEntries([...workload.matchAll(/"([^"]+)":\s*([\d.]+)/g)].map(m=>[m[1],Number(m[2])*1000]));
function tat(order){let time=0,sum=0;for(const id of order){time+=serviceMs[id];sum+=time;}return sum/order.length;}
const c=evidence.cfs;
if(Math.abs(tat(c.fifo_order)-c.fifo_tat_ms)>1e-6||Math.abs(tat(c.cfs_order)-c.cfs_tat_ms)>1e-6)throw Error('Source mismatch');
if(!evidence.checks.every(x=>x.ok))throw Error('Component checks not all passed');
await fs.writeFile(path.join(dir,'os-data.json'),JSON.stringify({source,workloadSource,serviceMs,fifoOrder:c.fifo_order,fairOrder:c.cfs_order,fifoTatMs:c.fifo_tat_ms,fairTatMs:c.cfs_tat_ms,checks:evidence.checks},null,2)+'\n');
console.log('OS evidence generated from original checks and workload; turnaround verified.');
