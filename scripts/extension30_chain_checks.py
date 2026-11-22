"""Public-state decomposition of the original questions, independent of model outcomes."""
def chain_checks(case, family):
 bg=case['public_background'];ts=bg['tasks'];ev=bg['events'];st=case['current_state']
 def item(field,question,value):return (field,question,value)
 if family=='admission-mem':
  return [item('ready_admission_memory_mb','Sum memory_mb ONLY for tasks whose state is READY. Exclude all other states. This is the memory demand of the jobs admitted in the original question.',sum(t['memory_mb'] for t in ts if t['state']=='READY'))]
 if family in ('release-admit','stage-handoff'):
  released=sum(t['memory_mb'] for t in ts if t['state']=='RUNNING')
  return [item('free_memory_after_release_mb','Compute free_memory_mb after the currently RUNNING predecessor finishes and releases its memory, before admitting its successor.',st['free_memory_mb']+released)]
 if family=='replica-memory':
  t=next(t for t in ts if 'replicas' in t)
  return [item('replica_admission_memory_mb','Compute the total memory required by all replicas of the task with a replicas field, before subtracting from current free memory.',t['memory_mb']*t['replicas'])]
 if family=='reserve-batch':
  reserve=next(e['memory_mb'] for e in ev if e.get('event')=='reservation')
  return [item('ready_batch_memory_mb','Sum the memory_mb of every READY batch job, excluding RUNNING and BLOCKED jobs.',sum(t['memory_mb'] for t in ts if t['state']=='READY')),item('free_memory_after_reservation_mb','Compute current free_memory_mb after setting aside the failover reservation, before admitting any READY job.',st['free_memory_mb']-reserve)]
 if family=='cpu-turnover':
  return [item('free_cpu_after_eviction','Compute current free_cpu_slots after evicting the RUNNING unhealthy task and releasing its CPU, before admitting the READY job.',st['free_cpu_slots']+sum(t['cpu'] for t in ts if t['state']=='RUNNING')),item('new_job_cpu_request','Read the CPU request of the READY replacement job that the original question admits.',sum(t['cpu'] for t in ts if t['state']=='READY'))]
 if family=='retry-wait':
  t=next(t for t in ts if 'max_retries' in t)
  return [item('remaining_retry_count','Compute remaining retries as max_retries minus the current attempt count for the retrying task.',t['max_retries']-t['attempt']),item('single_backoff_seconds','Read the per-retry backoff_s from the retrying task and confirm against retry_scheduled evidence.',t['backoff_s'])]
 if family=='timeout-buffer':
  beat=next(e for e in ev if e.get('event')=='heartbeat');buf=next(e['buffer_s'] for e in ev if e.get('event')=='reserved_buffer')
  return [item('timeout_remaining_before_buffer_seconds','Compute timeout_s minus elapsed_s from the most recent heartbeat, before deducting the retry reserve.',beat['timeout_s']-beat['elapsed_s']),item('reserved_retry_buffer_seconds','Read the buffer_s reserved by the retry policy; do not subtract it a second time.',buf)]
 if family=='transfer':
  link=next(e for e in ev if 'dataset_mb' in e)
  return [item('transfer_without_handshake_seconds','Compute dataset_mb divided by throughput_mb_s, excluding fixed handshake overhead.',link['dataset_mb']//link['throughput_mb_s']),item('handshake_overhead_seconds','Read handshake_overhead_s from the link profile, to be added once to transfer duration.',link['handshake_overhead_s'])]
 if family=='latest-start':
  t=next(t for t in ts if 'hard_deadline_s' in t);now=next(e['now_s'] for e in ev if 'now_s' in e)
  return [item('latest_start_clock_seconds','Compute hard_deadline_s minus remaining_work_s: the latest absolute clock time at which the job can start.',t['hard_deadline_s']-t['remaining_work_s']),item('time_until_deadline_seconds','Compute hard_deadline_s minus current now_s, before deducting remaining_work_s.',t['hard_deadline_s']-now)]
 if family=='queue-wait':
  order=next(e['ahead_of_preview'] for e in ev if 'ahead_of_preview' in e);jobs={t['id']:t for t in ts};first=order[:2];rest=order[2:]
  return [item('first_two_queued_duration_seconds','Sum estimated_duration_s of the first two jobs in the explicit ahead_of_preview order.',sum(jobs[x]['estimated_duration_s'] for x in first)),item('remaining_queued_duration_seconds','Sum estimated_duration_s of jobs after the first two in ahead_of_preview. Include only jobs ahead of preview.',sum(jobs[x]['estimated_duration_s'] for x in rest))]
 if family=='gpu-waves':
  return [item('queued_gpu_slot_demand','Sum gpu requests of all jobs named by the GPU backlog event, to check the number of 2-slot waves.',sum(t['gpu'] for t in ts))]
 if family=='batch-makespan':
  jobs={t['id']:t for t in ts};pip=next(e for e in ev if e.get('pipeline')=='A')
  return [item('pipeline_a_duration_seconds','Sum duration_s of the stages listed in the pipeline A event, in serial order.',sum(jobs[x]['duration_s'] for x in pip['order']))]
 if family=='critical-path':
  jobs={t['id']:t for t in ts};ends=[t for t in ts if not any(t['id'] in x['depends_on'] for x in ts)]
  def duration(t):return t['duration_s']+max([duration(jobs[x]) for x in t['depends_on']] or [0])
  assert len(ends)==2
  return [item('dependency_chain_one_seconds',f"Compute the full dependency-chain duration ending at {ends[0]['id']}, including completed ancestors.",duration(ends[0])),item('dependency_chain_two_seconds',f"Compute the full dependency-chain duration ending at {ends[1]['id']}, including completed ancestors.",duration(ends[1]))]
 raise ValueError(family)
