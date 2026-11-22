"""Public-input-only arithmetic for offline validation of parallel question copies."""
import copy
SCALED_KEYS={'cpu_slots','gpu_slots','memory_mb','cpu','gpu','free_cpu_slots','free_gpu_slots','free_memory_mb','duration_s','estimated_duration_s','remaining_work_s','hard_deadline_s','now_s','elapsed_s','timeout_s','buffer_s','backoff_s','handshake_overhead_s','dataset_mb'}
def scaled_copy(original,factor):
 def walk(v,key=None):
  if isinstance(v,dict):return {k:walk(x,k) for k,x in v.items()}
  if isinstance(v,list):return [walk(x) for x in v]
  if isinstance(v,(int,float)) and not isinstance(v,bool) and key in SCALED_KEYS:return v*factor
  return copy.deepcopy(v)
 packet=walk({'public_background':original['public_background'],'current_state':original['current_state']})
 bg=packet['public_background'];family=original['id'].removeprefix('os-c2c-probe3c-').rsplit('-',1)[0]
 if family=='gpu-waves':
  old=original['public_background']['runtime']['gpu_slots'];bg['pending_decision']['question']=bg['pending_decision']['question'].replace(f'{old} gpu_slots',f"{bg['runtime']['gpu_slots']} gpu_slots")
 if family=='latest-start':
  job=next(t for t in bg['tasks'] if 'hard_deadline_s' in t);now=next(e['now_s'] for e in bg['events'] if 'now_s' in e)
  bg['pending_decision']['question']=f"The wall clock is at {now}s and {job['id']} still needs {job['remaining_work_s']}s of work before its hard deadline at {job['hard_deadline_s']}s. How many seconds of slack remain before {job['id']} must start?"
 return packet
def solve(packet,family):
 bg=packet['public_background'];ts=bg['tasks'];es=bg['events'];st=packet['current_state'];ready=[t for t in ts if t['state']=='READY'];running=[t for t in ts if t['state']=='RUNNING']
 if family=='admission-mem':return st['free_memory_mb']-sum(t['memory_mb'] for t in ready)
 if family in ('release-admit','stage-handoff'):return st['free_memory_mb']+sum(t['memory_mb'] for t in running)-sum(t['memory_mb'] for t in ready)
 if family=='replica-memory':
  t=next(t for t in ts if 'replicas' in t);return st['free_memory_mb']-t['memory_mb']*t['replicas']
 if family=='reserve-batch':return st['free_memory_mb']-next(e['memory_mb'] for e in es if e.get('event')=='reservation')-sum(t['memory_mb'] for t in ready)
 if family=='cpu-turnover':return st['free_cpu_slots']+sum(t['cpu'] for t in running)-sum(t['cpu'] for t in ready)
 if family=='retry-wait':
  t=next(t for t in ts if 'max_retries' in t);return (t['max_retries']-t['attempt'])*t['backoff_s']
 if family=='timeout-buffer':
  h=next(e for e in es if e.get('event')=='heartbeat');return h['timeout_s']-h['elapsed_s']-next(e['buffer_s'] for e in es if e.get('event')=='reserved_buffer')
 if family=='transfer':
  e=next(e for e in es if 'dataset_mb' in e);assert e['dataset_mb']%e['throughput_mb_s']==0;return e['dataset_mb']//e['throughput_mb_s']+e['handshake_overhead_s']
 if family=='latest-start':
  t=next(t for t in ts if 'hard_deadline_s' in t);return t['hard_deadline_s']-t['remaining_work_s']-next(e['now_s'] for e in es if 'now_s' in e)
 if family=='queue-wait':
  e=next(e for e in es if e.get('event')=='order');ids=next(v for k,v in e.items() if k.startswith('ahead_of_'));jobs={t['id']:t for t in ts};return sum(jobs[x]['estimated_duration_s'] for x in ids)
 if family=='batch-makespan':
  jobs={t['id']:t for t in ts};return max(sum(jobs[x]['duration_s'] for x in e['order']) for e in es if 'pipeline' in e)
 if family=='critical-path':
  jobs={t['id']:t for t in ts}
  def longest(t):return t['duration_s']+max([longest(jobs[x]) for x in t['depends_on']] or [0])
  return max(longest(t) for t in ts if t['state'] not in ('SUCCEEDED','DONE'))
 if family=='gpu-waves':
  jobs={t['id']:t for t in ts};ids=next(e['jobs'] for e in es if e.get('queue')=='gpu');demands=sorted([jobs[x]['gpu'] for x in ids],reverse=True);cap=bg['runtime']['gpu_slots'];assert all(0<x<=cap for x in demands);best=len(demands)
  def pack(i,bins):
   nonlocal best
   if len(bins)>=best:return
   if i==len(demands):best=len(bins);return
   seen=set()
   for j,load in enumerate(bins):
    if load in seen or load+demands[i]>cap:continue
    seen.add(load);bins[j]+=demands[i];pack(i+1,bins);bins[j]-=demands[i]
   pack(i+1,bins+[demands[i]])
  pack(0,[]);return best
 raise ValueError(family)
