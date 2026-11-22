"""Validate os-classics scenarios: simulate from public_background, assert probe answers.

This mirrors the host-side _simulate_os_classics that will be added to c2c_core.py.
Run: python scripts/check_os_classics_scenarios.py
"""
import json
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "c2c-core-os"


def sim_fcfs_sjf(jobs):
    """jobs: list of dicts with id, arrival_seconds, burst_seconds.
    Returns dict with fcfs_total_turnaround, sjf_total_turnaround, sjf_total_waiting."""
    order_fcfs = sorted(jobs, key=lambda j: (j["arrival_seconds"], j["id"]))
    t, total_t = 0, 0
    for j in order_fcfs:
        t = max(t, j["arrival_seconds"]) + j["burst_seconds"]
        total_t += t - j["arrival_seconds"]
    fcfs_tt = total_t

    done, t, total_t, total_w = set(), 0, 0, 0
    while len(done) < len(jobs):
        ready = [j for j in jobs if j["id"] not in done and j["arrival_seconds"] <= t]
        if not ready:
            t = min(j["arrival_seconds"] for j in jobs if j["id"] not in done)
            continue
        nxt = min(ready, key=lambda j: (j["burst_seconds"], j["arrival_seconds"], j["id"]))
        t = max(t, nxt["arrival_seconds"]) + nxt["burst_seconds"]
        total_t += t - nxt["arrival_seconds"]
        total_w += t - nxt["arrival_seconds"] - nxt["burst_seconds"]
        done.add(nxt["id"])
    return {"fcfs_total_turnaround_seconds": fcfs_tt,
            "sjf_total_turnaround_seconds": total_t,
            "sjf_total_waiting_seconds": total_w}


def sim_banker(resource, procs, request=None):
    """request: (proc_id, units) or None. Returns safe_before_request / p1_request_one_grant."""
    def safe(avail, alloc, need):
        work, finished = avail, set()
        while len(finished) < len(procs):
            prog = False
            for p in procs:
                if p["id"] in finished:
                    continue
                if need[p["id"]] <= work:
                    work += alloc[p["id"]]
                    finished.add(p["id"])
                    prog = True
            if not prog:
                return 0
        return 1

    alloc = {p["id"]: p["allocated_units"] for p in procs}
    maxu = {p["id"]: p["maximum_units"] for p in procs}
    need = {pid: maxu[pid] - alloc[pid] for pid in alloc}
    avail = resource["available_units"]
    before = safe(avail, alloc, dict(need))
    out = {"safe_before_request": before}
    if request:
        pid, units = request
        grant = int(safe(avail - units,
                         {**alloc, pid: alloc[pid] + units},
                         {**need, pid: need[pid] - units}))
        out["request_grant"] = grant
    return out


def sim_paging(refs, frames):
    fifo_q, fifo_set, fifo_faults = [], set(), 0
    for pg in refs:
        if pg not in fifo_set:
            fifo_faults += 1
            if len(fifo_q) >= frames:
                fifo_set.discard(fifo_q.pop(0))
            fifo_q.append(pg)
            fifo_set.add(pg)
    lru_age, lru_faults = {}, 0
    for i, pg in enumerate(refs):
        if pg not in lru_age:
            lru_faults += 1
            if len(lru_age) >= frames:
                evict = min(lru_age, key=lambda k: lru_age[k])
                del lru_age[evict]
        lru_age[pg] = i
    return {"fifo_faults": fifo_faults, "lru_faults": lru_faults}


def sim_fault(bg):
    evs = bg["agent_events"]
    idx = next(i for i, e in enumerate(evs)
               if e["exit_code"] == 137 and e["peak_rss_mb"] > bg["memory_max_mb"])
    survived = sum(1 for e in evs if e["exit_code"] == 0)
    return {"oom_agent_index": idx, "survived_siblings": survived}


def expected_for(case):
    fam = case["probe"]["family"]
    bg = case["public_background"]
    if fam == "os_classics_sched":
        return sim_fcfs_sjf(bg["tasks"])
    if fam == "os_classics_schedstep":
        jobs = {j["id"]: j for j in bg["tasks"]}
        out, prev = {}, 0
        for jid in bg["decided_dispatch_order"]:
            job = jobs[jid]
            start = max(prev, job["arrival_seconds"])
            prev = start + job["burst_seconds"]
            out[f"completion_{jid.lower()}_seconds"] = prev
        return out
    if fam == "os_classics_banker":
        req_ev = next(e for e in bg["events"] if e.get("event") == "request_arrived")
        return sim_banker(bg["resource"], bg["tasks"],
                          (req_ev["process"], req_ev["requested_units"]))
    if fam == "os_classics_paging":
        return sim_paging(bg["reference_string"], bg["frames"])
    if fam == "os_classics_fault":
        return sim_fault(bg)
    raise ValueError(fam)


def main():
    failures = 0
    for path in sorted(DATA.glob("public-osclassics-*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            sim = expected_for(case)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {path.name}: simulator error {exc!r}")
            failures += 1
            continue
        probe = case["probe"]
        probe_map = dict(zip(probe["intended_questions"], probe["expected_answers"]))
        # probe stores questions/answers positionally; match by derivation order is
        # fragile, so instead re-derive the field list from pending_decisions.
        fields = [d["field"] for d in case["public_background"]["pending_decisions"]]
        expected = probe["expected_answers"]
        for field, want in zip(fields, expected):
            if "request" in field and "grant" in field:
                got = sim.get("request_grant")
            else:
                got = sim.get(field)
            status = "ok" if got == want else "MISMATCH"
            if got != want:
                failures += 1
            print(f"[{status}] {path.name} {field}: sim={got} probe={want}")
        assert len(fields) == len(expected), f"{path.name}: field/answer count mismatch"
    if failures:
        print(f"\n{failures} failure(s)")
        sys.exit(1)
    print("\nall os-classics scenarios self-consistent")


if __name__ == "__main__":
    main()
