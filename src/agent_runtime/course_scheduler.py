"""Non-preemptive selection at common runnable-stage boundaries.

FIFO uses the latest READY enqueue sequence. vservice uses measured cumulative
service seconds divided by weight; waiting is not service. This is not Linux
vruntime and does not preempt an in-flight GPU generation.
"""


def pick_ready(acbs,policy):
    if policy not in {'fifo','vservice'}:
        raise ValueError('unknown scheduling policy')
    ready=[a for a in acbs if a['state']=='READY']
    if any(a.get('weight',1.0)<=0 for a in ready):
        raise ValueError('vservice weight must be positive')
    if not ready:
        return None
    def fifo(a):
        return (a['ready_sequence'],a['id'])
    if policy=='fifo':
        return min(ready,key=fifo)
    if policy=='vservice':
        return min(ready,key=lambda a:(a.get('virtual_service',0.0)/a.get('weight',1.0),*fifo(a)))
    raise ValueError('unknown scheduling policy')
