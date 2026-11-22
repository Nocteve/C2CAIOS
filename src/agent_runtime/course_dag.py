"""Exact public-job prerequisites and delivered Agent-artifact dependencies."""


def dependency_status(task,agents,jobs):
    blocked=[];invalid=[]
    for dep in task['depends_on']:
        if dep.startswith('agent:'):
            target=dep[6:]
            if target not in agents:invalid.append(dep)
            elif agents[target]['state'] not in {'REPORTED','SUCCEEDED'}:blocked.append(dep)
        else:
            target=dep[4:] if dep.startswith('job:') else dep
            if target not in jobs:invalid.append(dep)
            elif jobs[target]['state']!='SUCCEEDED':blocked.append(dep)
    return blocked,invalid


def has_agent_cycle(node,agents):
    def reaches(current,visited):
        if current==node:return True
        if current in visited or current not in agents:return False
        visited.add(current)
        task=agents[current].get('task',agents[current])
        return any(reaches(d[6:],visited) for d in task['depends_on'] if d.startswith('agent:'))
    task=agents[node].get('task',agents[node])
    return any(reaches(d[6:],set()) for d in task['depends_on'] if d.startswith('agent:'))
