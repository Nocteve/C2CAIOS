"""Public-state scheduling proposal validation; never supplies a repaired answer.

The business workload is a snapshot. Agent dependencies are a separate namespace.
This module does not run, complete or repair the snapshot's business programs.
"""
from copy import deepcopy

FIELDS = {"start", "wait", "monitor", "free_after"}
TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED"}


def check_plan(case, proposal, *, complete=False):
    errors = []
    if not isinstance(proposal, dict) or not proposal:
        return {"accepted": False, "errors": ["report must be a nonempty plan object"]}
    unknown = set(proposal) - FIELDS
    if unknown:
        errors.append("unknown plan fields: " + ",".join(sorted(unknown)))
    if complete and FIELDS - set(proposal):
        errors.append("incomplete plan fields: " + ",".join(sorted(FIELDS - set(proposal))))
    jobs = {x["id"]: x for x in case["public_background"]["tasks"]}
    state = case["current_state"]
    start = proposal.get("start", [])
    if not isinstance(start, list) or any(not isinstance(x, str) for x in start):
        errors.append("start must be a list of job IDs")
        start = []
    if len(start) != len(set(start)):
        errors.append("duplicate start job")
    remaining = dict(cpu=state["free_cpu_slots"], gpu=state["free_gpu_slots"], memory_mb=state["free_memory_mb"])
    for jid in start:
        job = jobs.get(jid)
        if not job:
            errors.append(f"start:{jid}: unknown public job")
            continue
        if job["state"] in TERMINAL | {"RUNNING"}:
            errors.append(f"start:{jid}: job is already active or terminal")
        if any(jobs[d]["state"] != "SUCCEEDED" for d in job["depends_on"]):
            errors.append(f"start:{jid}: unresolved dependency")
        for key in remaining:
            remaining[key] -= job[key]
    if any(v < 0 for v in remaining.values()):
        errors.append("start exceeds snapshot free capacity")
    if len(start) + len(state["active_tasks"]) > case["public_background"]["runtime"]["max_parallel_tasks"]:
        errors.append("start exceeds snapshot parallelism limit")
    if "free_after" in proposal:
        if "start" not in proposal:
            errors.append("free_after requires an explicit start in the same report")
        actual = proposal["free_after"]
        if (not isinstance(actual, dict) or set(actual) != set(remaining)
                or any(type(v) is not int for v in actual.values()) or actual != remaining):
            errors.append("free_after contradicts proposed start and public resource accounting")
    expected_wait = {jid: [d for d in j["depends_on"] if jobs[d]["state"] != "SUCCEEDED"]
                     for jid, j in jobs.items() if j["state"] not in TERMINAL | {"RUNNING"}
                     and any(jobs[d]["state"] != "SUCCEEDED" for d in j["depends_on"])}
    if "wait" in proposal:
        wait = proposal["wait"]
        if not isinstance(wait, dict):
            errors.append("wait must map public job IDs to unresolved dependency IDs")
        else:
            for jid, deps in wait.items():
                if (not isinstance(deps, list) or any(not isinstance(d, str) for d in deps)
                        or len(deps) != len(set(deps)) or jid not in expected_wait
                        or set(deps) != set(expected_wait[jid])):
                    errors.append(f"wait:{jid}: dependency claim contradicts public snapshot")
            if complete and set(wait) != set(expected_wait):
                errors.append("wait omits or adds dependency-blocked public jobs")
    expected_monitor = {}
    for event in case["public_background"]["events"]:
        if event.get("task") in state["active_tasks"] and "timeout_s" in event and "elapsed_s" in event:
            expected_monitor[event["task"]] = max(0, event["timeout_s"] - event["elapsed_s"])
    if "monitor" in proposal:
        monitor = proposal["monitor"]
        if not isinstance(monitor, dict):
            errors.append("monitor must map active job IDs to snapshot remaining seconds")
        else:
            for jid, value in monitor.items():
                if type(value) is not int or jid not in expected_monitor or value != expected_monitor[jid]:
                    errors.append(f"monitor:{jid}: watchdog claim contradicts public snapshot")
            if complete and set(monitor) != set(expected_monitor):
                errors.append("monitor omits or adds snapshot watchdog obligations")
    if complete and not start:
        feasible = any(j["state"] not in TERMINAL | {"RUNNING"}
                       and all(jobs[d]["state"] == "SUCCEEDED" for d in j["depends_on"])
                       and j["cpu"] <= state["free_cpu_slots"] and j["gpu"] <= state["free_gpu_slots"]
                       and j["memory_mb"] <= state["free_memory_mb"]
                       for j in jobs.values())
        if feasible and len(state["active_tasks"]) < case["public_background"]["runtime"]["max_parallel_tasks"]:
            errors.append("plan provides no schedulable next step despite available work")
    return {"accepted": not errors, "errors": errors, "public_state_derived": True,
            "complete": complete, "host_generated_repair": False}


def merge_proposal(case, current, report):
    """Explicit Planner acceptance only. Conflicting claims require a new report."""
    check = check_plan(case, report)
    if not check["accepted"]:
        return current, check
    merged = deepcopy(current)
    for key, value in report.items():
        if key in {"wait", "monitor"} and key in merged:
            for jid, claim in value.items():
                if jid in merged[key] and merged[key][jid] != claim:
                    return current, {"accepted": False, "errors": [f"conflicting accepted field:{key}:{jid}"], "host_generated_repair": False}
                merged[key][jid] = deepcopy(claim)
            continue
        if key in merged and merged[key] != value:
            return current, {"accepted": False, "errors": [f"conflicting accepted field:{key}; submit a complete replacement report"], "host_generated_repair": False}
        merged[key] = deepcopy(value)
    check = check_plan(case, merged)
    return (merged if check["accepted"] else current), check


PLAN_INTERFACE = {
    "start": "list of public job IDs to admit now; any feasible useful selection is allowed",
    "wait": "map each dependency-blocked public job to its exact currently unsatisfied dependency IDs",
    "monitor": "map each active job with a timeout event to integer remaining seconds AT THE SNAPSHOT (not wall-clock now)",
    "free_after": "{cpu:integer,gpu:integer,memory_mb:integer} after this report's explicit start list",
    "partial_reports": "nonempty subsets allowed; free_after requires start in the same report; final plan needs all four fields. Explicit acceptance unions distinct wait/monitor job keys; conflicting claims are rejected.",
    "scope": "Propose a scheduling plan; do not claim the public business jobs were executed. Report action payload must be a JSON object with these fields.",
}
