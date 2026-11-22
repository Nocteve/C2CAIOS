"""OS-methodology scheduler evidence: CFS fairness, deadlock detection,
banker's admission, and the simulated cgroup controller.

All tests are pure orchestration (fake backends, deterministic sleeps) — the
point is mechanism evidence that survives on any machine, GPU or not.

Design notes (honesty first):
- T1's baseline is the REAL baseline scheduler (acb.ACBScheduler, static
  priority FIFO) — the thing CFS is compared against is the shipped system.
- T3's backend models parallel hold-and-wait semantics inside the serial
  harness: a task sees every admitted (RUNNING) task's reservation as held.
  Under "fit" admission both tasks hold 8/10 and each then wants +6 -> the
  unsafe state blocks forever; banker holds the second task in READY. The
  serial production runtime cannot resource-deadlock by itself; the banker
  is the admission policy for the parallel execution mode the PPT targets,
  and the deadlock shape is demonstrated, not faked.
"""
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent_runtime.acb import ACBScheduler, AgentControlBlock
from agent_runtime.os_sched import FairScheduler
from agent_runtime.resource_control import (ControlledBackend, ResourceController,
                                            SimulatedCgroupFs, SliceQuota, TaskQuota)

OUT = Path("artifacts/os-sched-evidence-20261019")
OUT.mkdir(parents=True, exist_ok=True)
results = []


def check(name, ok, detail):
    results.append({"name": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name} :: {detail}")


class SerialBackend:
    """Deterministic serial backend: service time = est_tokens ms."""

    gpu_budget_bytes = 0

    def heartbeat(self, acb):
        return time.perf_counter()

    def execute(self, acb):
        secs = (acb.est_tokens or 1) / 1000.0
        time.sleep(secs)
        return {"artifact": "ok", "gate": {"accepted": True},
                "error": None, "wait_states": {"WAIT_MODEL": secs}}


def order_of(report_events):
    return [e["task"] for e in report_events if e.get("event") == "done"]


def turnaround(order, service):
    t, total = 0.0, 0.0
    for tid in order:
        t += service[tid]
        total += t
    return total / len(order)


# --- T1: CFS (min-vruntime) vs the shipped FIFO scheduler -------------------
service = {"big1": 0.400, "big2": 0.400, "s1": 0.030, "s2": 0.030,
           "s3": 0.030, "s4": 0.030, "s5": 0.030, "s6": 0.030}
submitted = ["big1", "big2", "s1", "s2", "s3", "s4", "s5", "s6"]  # big tasks first

fifo_events = []
fifo = ACBScheduler(SerialBackend(), emit=fifo_events.append)
for i, tid in enumerate(submitted):
    fifo.submit(AgentControlBlock(
        task_id=tid, command=tid, priority=float(len(submitted) - i),
        est_tokens=int(service[tid] * 1000)))
fifo_report = fifo.run()
fifo_order = order_of(fifo_events)

cfs_events = []
cfs = FairScheduler(SerialBackend(), emit=cfs_events.append)
for i, tid in enumerate(submitted):
    cfs.submit(AgentControlBlock(
        task_id=tid, command=tid, priority=float(len(submitted) - i),
        est_tokens=int(service[tid] * 1000)))
cfs_report = cfs.run()
cfs_order = order_of(cfs_events)

fifo_tat = turnaround(fifo_order, service)
cfs_tat = turnaround(cfs_order, service)
small = {f"s{i}": service[f"s{i}"] for i in range(1, 7)}


def mean_waiting(order, svc):
    """Queueing delay = turnaround - own service (the CFS figure of merit:
    how long a ready task waits behind others)."""
    t, total = 0.0, 0.0
    for tid in order:
        t += svc[tid]
        total += t - svc[tid]
    return total / len(order)


fifo_small = mean_waiting(fifo_order, service)
cfs_small = mean_waiting(cfs_order, service)
check("cfs-vs-fifo-mean-turnaround", cfs_tat < fifo_tat,
      f"FIFO {fifo_tat * 1000:.0f}ms vs CFS {cfs_tat * 1000:.0f}ms | "
      f"FIFO order {fifo_order} | CFS order {cfs_order}")
check("cfs-small-task-waiting", cfs_small < fifo_small,
      f"small-task mean waiting FIFO {fifo_small * 1000:.0f}ms vs CFS {cfs_small * 1000:.0f}ms")
check("cfs-invariants", all(i["ok"] for i in cfs_report["invariants"]),
      f"CFS I1/I2 all ok, done {cfs_report['done']}/{cfs_report['total']}")

# --- T2: deadlock detection (wait-for cycle) --------------------------------
det = FairScheduler(SerialBackend(), emit=lambda r: None)
det.submit(AgentControlBlock(task_id="t1", command="c1", depends_on=("t2",)))
det.submit(AgentControlBlock(task_id="t2", command="c2", depends_on=("t1",)))
det_report = det.run()
cycles = det_report["scheduling"]["deadlocks"]
check("deadlock-cycle-detected",
      len(cycles) == 1 and cycles[0]["cycle"] == ["t1", "t2", "t1"],
      f"cycles={[c['cycle'] for c in cycles]}")
check("deadlock-no-execution",
      det_report["tasks"]["t1"]["state"] == "FAILED"
      and det_report["tasks"]["t2"]["state"] == "FAILED"
      and all(t["attempts"] == 0 for t in det_report["tasks"].values()),
      "both tasks failed fast with zero executions (EDEADLK semantics)")

# --- T3: banker's admission under hold-and-wait ------------------------------
class HoldAndWaitBackend:
    """Models parallel hold-and-wait inside the serial harness: every admitted
    (RUNNING) task's reservation is treated as held while a task executes;
    a mid-run burst request blocks when held+burst exceeds the budget —
    exactly the deadlock a parallel runtime would enter."""

    gpu_budget_bytes = 10

    def __init__(self):
        self.sched = None

    def heartbeat(self, acb):
        return time.perf_counter()

    def execute(self, acb):
        held = sum(a.gpu_bytes_est for a in self.sched.acbs.values()
                   if a.state in ("RUNNING", "WAIT_MODEL", "WAIT_FUSION"))
        burst = int(getattr(acb, "burst_bytes", 0))
        if held + burst > self.gpu_budget_bytes:
            while True:                      # the unsafe state: waits forever
                time.sleep(3600)
        time.sleep(0.01)
        return {"artifact": "ok", "gate": {"accepted": True}, "error": None,
                "wait_states": {"WAIT_MODEL": 0.01}}


def run_banker(admission):
    backend = HoldAndWaitBackend()
    sched = FairScheduler(backend, admission=admission)
    backend.sched = sched
    sched.submit(AgentControlBlock(task_id="A", command="a", gpu_bytes_est=4))
    sched.submit(AgentControlBlock(task_id="B", command="b", gpu_bytes_est=4))
    for acb in sched.acbs.values():
        acb.burst_bytes = 6               # each will request +6 mid-run
    box = {}
    th = threading.Thread(target=lambda: box.update(report=sched.run()),
                          daemon=True)
    th.start()
    th.join(timeout=8)
    return th.is_alive(), box.get("report")


naive_alive, naive_report = run_banker("fit")
banker_alive, banker_report = run_banker("banker")
check("banker-naive-deadlocks", naive_alive and naive_report is None,
      "fit-admission: both tasks hold 8/10, each wants +6 -> deadlock (leashed by the 8s test thread)")
check("banker-safe-state-prevents",
      (not banker_alive) and banker_report is not None
      and banker_report["done"] == 2,
      f"banker run finished with done={banker_report['done'] if banker_report else '?'} "
      "(B held in READY until A released; then B completed)")

# --- T4: simulated cgroup controller -----------------------------------------
fs = SimulatedCgroupFs(OUT / "cgroups")
events = []
controller = ResourceController(
    fs, task_quota=TaskQuota(tokens=50, wall_s=5.0, memory_mib=512),
    slice_quota=SliceQuota(pids_max=4),
    emit=events.append)


class TokenEaterBackend:
    gpu_budget_bytes = 0

    def heartbeat(self, acb):
        return time.perf_counter()

    def execute(self, acb):
        if acb.task_id.startswith("hung"):
            return {"artifact": "x" * 10,
                    "record": {"output_tokens": 80, "peak_incremental_mib": 64,
                               "service_seconds": 6.0},
                    "gate": {"accepted": True}, "error": None,
                    "wait_states": {"WAIT_MODEL": 6.0}}
        return {"artifact": "ok",
                "record": {"output_tokens": 20, "peak_incremental_mib": 100,
                           "service_seconds": 1.0},
                "gate": {"accepted": True}, "error": None,
                "wait_states": {"WAIT_MODEL": 1.0}}


ctl_events = []
ctl_sched = FairScheduler(ControlledBackend(TokenEaterBackend(), controller),
                          emit=ctl_events.append)
ctl_sched.submit(AgentControlBlock(task_id="ok1", command="a"))
ctl_sched.submit(AgentControlBlock(task_id="hung1", command="b"))
ctl_report = ctl_sched.run()
violations = controller.violations
check("cgroup-quota-kill", any(v["kind"] == "tokens" for v in violations),
      f"violations={[(v['task'], v['kind']) for v in violations]} | states: "
      + ", ".join(f"{t['task_id']}:{t['state']}" for t in ctl_report["tasks"].values()))
cg_dir = fs.slice / "task-hung1.scope"
mem_events = (cg_dir / "memory.events").read_text()
check("cgroup-files-live",
      cg_dir.exists() and "oom_kill" in mem_events
      and (cg_dir / "tokens.current").read_text().strip() == "80",
      f"hung1 tokens.current=80, memory.events tail={mem_events.splitlines()[-2:]}")
check("cgroup-parent-survives",
      ctl_report["wall_seconds"] < 30 and all(i["ok"] for i in ctl_report["invariants"]),
      "runtime survived the quota kill; invariants intact")

# fork-fail semantics: slice pids.max denies the 5th live task
phantoms = [AgentControlBlock(task_id=f"p{i}", command="x") for i in range(5)]
admits = [controller.try_admit(a) for a in phantoms]
check("cgroup-fork-fail", all(a[0] for a in admits[:4]) and (not admits[4][0])
      and admits[4][1],
      f"4 admits ok, 5th denied: {admits[4][1]!r} (cgroup fork-fail semantics)")

(OUT / "checks.json").write_text(json.dumps(
    {"checks": results,
     "cfs": {"fifo_order": fifo_order, "cfs_order": cfs_order,
             "fifo_tat_ms": fifo_tat * 1000, "cfs_tat_ms": cfs_tat * 1000,
             "fifo_small_ms": fifo_small * 1000, "cfs_small_ms": cfs_small * 1000},
     "deadlock_cycles": [c["cycle"] for c in cycles],
     "quota_violations": [(v["task"], v["kind"]) for v in violations]},
    ensure_ascii=False, indent=1), encoding="utf-8")
print("\nALL PASS" if all(r["ok"] for r in results) else "\nSOME FAILED")
sys.exit(0 if all(r["ok"] for r in results) else 1)
