"""Agent Control Block (ACB) and scheduler — OS-style lifecycle for agent tasks.

Design: docs/b1-acb-design-2026-10-18.md (proposal PPT p6/7/8/9). This module is
pure orchestration: no torch, no model I/O. A backend executes admitted tasks
and reports heartbeats, which keeps the state machine unit-testable with a
fake backend and reusable on the real GPU path.

States (LLM-specific waits are first-class, PPT p6):
  READY            deps not all DONE, waiting for admission
  RUNNING          admitted, backend owns it
  WAIT_MODEL       inside the backend: holding a GPU slot for generation
  WAIT_FUSION      inside the backend: waiting for fused cache publish/clone
  WAIT_DEPENDENCY  admitted but a declared dependency is not DONE (never RUNNING)
  DONE             gate accepted
  FAILED           gate rejected / heartbeat timeout / backend error
  REPAIR_PENDING   failure recorded, repair task creation in flight

Every transition goes through ACB.transition() and is appended to acb.events
with a monotonic timestamp; the scheduler replays events for invariant checks.
"""

from dataclasses import dataclass, field
from time import perf_counter

STATES = ("READY", "RUNNING", "WAIT_MODEL", "WAIT_DEPENDENCY", "WAIT_FUSION",
          "DONE", "FAILED", "REPAIR_PENDING")

TERMINAL = ("DONE", "FAILED")


@dataclass
class AgentControlBlock:
    task_id: str
    command: str
    deliverable: str = ""
    depends_on: tuple = ()
    priority: float = 0.0
    gpu_bytes_est: int = 0
    est_tokens: int = 0
    max_attempts: int = 2          # 1 original + 1 repair
    heartbeat_timeout: float = 120.0
    repair_of: str | None = None   # set on repair tasks
    state: str = "READY"
    attempts: int = 0
    last_heartbeat: float = 0.0
    artifact: str | None = None
    gate: dict | None = None
    error: str | None = None
    events: list = field(default_factory=list)

    def transition(self, new_state, cause=""):
        if new_state not in STATES:
            raise ValueError(f"unknown state {new_state}")
        self.events.append({"t": perf_counter(), "from": self.state,
                            "to": new_state, "cause": cause})
        self.state = new_state

    def wait_seconds(self, *states):
        """Time spent in the given wait states. Backend-measured waits are
        taken from measured_seconds markers; scheduler-native waits (READY /
        WAIT_DEPENDENCY) are replayed from the event log."""
        want = set(states)
        measured = sum(e.get("measured_seconds", 0.0)
                       for e in self.events if e.get("to") in want)
        total = 0.0
        open_from = None
        for e in self.events:
            if "measured_seconds" in e:
                continue  # already counted
            if e["from"] in want and e["to"] not in want:
                total += e["t"] - open_from if open_from else 0.0
                open_from = None
            if e["from"] not in want and e["to"] in want:
                open_from = e["t"]
        if open_from is not None and self.state in want:
            total += perf_counter() - open_from
        return total + measured


class SchedulerError(RuntimeError):
    pass


class ACBScheduler:
    """Non-preemptive admission scheduler over a DAG with runtime growth.

    backend protocol (duck-typed):
      backend.execute(acb) -> dict with keys:
        "artifact": str|None      worker raw output
        "gate": dict|None         application gate result (accepted bool)
        "error": str|None         backend failure, if any
        "wait_states": dict       seconds spent in WAIT_MODEL/WAIT_FUSION
      backend.heartbeat(acb) -> float|None   last beat monotonic time
      backend.gpu_budget_bytes -> int
    repair_policy(acb) -> command str|None   None = no repair (attempts exhausted)
    """

    def __init__(self, backend, repair_policy=None, emit=None):
        self.backend = backend
        self.repair_policy = repair_policy
        self.emit = emit or (lambda rec: None)
        self.acbs = {}
        self.invariants = []

    def submit(self, acb):
        if acb.task_id in self.acbs:
            raise SchedulerError(f"duplicate task {acb.task_id}")
        self.acbs[acb.task_id] = acb
        acb.transition("READY", "submitted")
        self.emit({"event": "submit", "task": acb.task_id})

    def _deps_done(self, acb):
        return all(self.acbs[d].state == "DONE" for d in acb.depends_on
                   if d in self.acbs)

    def _deps_failed(self, acb):
        return any(self.acbs[d].state in TERMINAL and self.acbs[d].state != "DONE"
                   for d in acb.depends_on if d in self.acbs)

    def run(self):
        started = perf_counter()
        budget = int(getattr(self.backend, "gpu_budget_bytes", 0) or 0)
        in_flight = {}
        admitted_bytes = 0
        while True:
            # 1. heartbeat supervision (PPT p8)
            for tid, acb in list(in_flight.items()):
                beat = self.backend.heartbeat(acb)
                acb.last_heartbeat = beat if beat is not None else acb.last_heartbeat
                if beat is None and acb.state in ("RUNNING", "WAIT_MODEL", "WAIT_FUSION"):
                    acb.transition("FAILED", "heartbeat_lost")
                    self.emit({"event": "heartbeat_lost", "task": tid})
                    in_flight.pop(tid)
                    admitted_bytes -= acb.gpu_bytes_est
                    self._after_terminal(acb)

            # 2. dependency bookkeeping: WAIT_DEPENDENCY <-> READY, fail on dead deps
            for acb in self.acbs.values():
                if acb.state == "WAIT_DEPENDENCY":
                    if self._deps_failed(acb):
                        acb.transition("FAILED", "dependency_failed")
                        self._after_terminal(acb)
                    elif self._deps_done(acb):
                        acb.transition("READY", "dependencies_done")
                elif acb.state == "READY" and not self._deps_done(acb):
                    acb.transition("WAIT_DEPENDENCY", "dependency_not_done")

            # 3. admission by priority under the GPU budget (PPT p7)
            ready = sorted((a for a in self.acbs.values() if a.state == "READY"
                            and a not in in_flight.values()),
                           key=lambda a: -a.priority)
            for acb in ready:
                if budget and admitted_bytes + acb.gpu_bytes_est > budget:
                    continue
                acb.transition("RUNNING", "admitted")
                acb.attempts += 1
                in_flight[acb.task_id] = acb
                admitted_bytes += acb.gpu_bytes_est
                self.emit({"event": "admit", "task": acb.task_id,
                           "admitted_bytes": admitted_bytes, "budget": budget})

            # 3b. pre-flight liveness: the backend must confirm the task is
            #     alive before it consumes a generation slot. In serial mode
            #     this is the effective heartbeat granularity (design doc).
            for tid, acb in list(in_flight.items()):
                if self.backend.heartbeat(acb) is None:
                    acb.transition("FAILED", "heartbeat_lost_preflight")
                    self.emit({"event": "heartbeat_lost", "task": tid})
                    in_flight.pop(tid)
                    admitted_bytes -= acb.gpu_bytes_est
                    self._after_terminal(acb)

            # 4. execute one task (serial execution: admission is accounted,
            #    execution stays serial per design doc section 4)
            if in_flight:
                tid, acb = next(iter(in_flight.items()))
                acb.transition("WAIT_MODEL", "backend_execute_begin")
                result = self.backend.execute(acb) or {}
                wait_states = result.get("wait_states") or {}
                for ws, secs in wait_states.items():
                    acb.events.append({"t": perf_counter(), "from": "RUNNING",
                                       "to": ws, "cause": f"measured:{secs:.6f}s",
                                       "measured_seconds": secs})
                acb.artifact = result.get("artifact")
                acb.gate = result.get("gate")
                acb.error = result.get("error")
                # return to RUNNING then settle terminal state
                acb.transition("RUNNING", "backend_execute_end")
                if acb.error:
                    acb.transition("FAILED", f"backend_error:{acb.error}")
                elif (acb.gate or {}).get("accepted"):
                    acb.transition("DONE", "gate_accepted")
                else:
                    acb.transition("FAILED",
                                   f"gate_rejected:{(acb.gate or {}).get('reason')}")
                in_flight.pop(tid)
                admitted_bytes -= acb.gpu_bytes_est
                self.emit({"event": acb.state.lower(), "task": tid,
                           "gate": acb.gate, "error": acb.error})
                self._after_terminal(acb)
                continue

            # 5. nothing running: done if no non-terminal tasks remain
            #    (REPAIR_PENDING is settled: its repair task is in the READY pool)
            pending = [a for a in self.acbs.values()
                       if a.state not in TERMINAL and a.state != "REPAIR_PENDING"]
            if not pending:
                break
            # READY tasks can still be admitted next iteration; a deadlock is
            # only when every pending task is blocked on dependencies.
            if all(a.state == "WAIT_DEPENDENCY" for a in pending):
                for a in pending:
                    if a.state != "FAILED":
                        a.transition("FAILED", "dependency_deadlock_or_missing")
                break
        report = self._report(perf_counter() - started)
        self._check_invariants()
        report["invariants"] = self.invariants
        return report

    def _after_terminal(self, acb):
        if acb.state != "FAILED":
            return
        if self.repair_policy is None:
            return
        if acb.attempts >= acb.max_attempts:
            return
        cmd = self.repair_policy(acb)
        if not cmd:
            return
        acb.transition("REPAIR_PENDING", "repair_scheduled")
        rid = f"{acb.task_id}.r{acb.attempts}"
        repair = AgentControlBlock(
            task_id=rid, command=cmd, deliverable=acb.deliverable,
            depends_on=(), priority=acb.priority + 1e6,  # repairs jump the queue
            gpu_bytes_est=acb.gpu_bytes_est, est_tokens=acb.est_tokens,
            max_attempts=1, repair_of=acb.task_id)
        self.submit(repair)
        self.emit({"event": "repair_submit", "task": rid, "of": acb.task_id})

    def _report(self, wall):
        by_state = {}
        for a in self.acbs.values():
            rec = {"task_id": a.task_id, "state": a.state, "attempts": a.attempts,
                   "repair_of": a.repair_of,
                   "wait_model_s": a.wait_seconds("WAIT_MODEL"),
                   "wait_fusion_s": a.wait_seconds("WAIT_FUSION"),
                   "wait_dependency_s": a.wait_seconds("WAIT_DEPENDENCY"),
                   "ready_s": a.wait_seconds("READY"),
                   "gate_accepted": bool((a.gate or {}).get("accepted"))}
            by_state[a.task_id] = rec
        done = sum(1 for a in self.acbs.values() if a.state == "DONE")
        return {"wall_seconds": wall, "tasks": by_state,
                "done": done, "total": len(self.acbs),
                "repair_count": sum(1 for a in self.acbs.values() if a.repair_of)}

    def _check_invariants(self):
        # I1: never RUNNING (or beyond) before depends_on all DONE — replay events
        for a in self.acbs.values():
            deps = a.depends_on
            if not deps:
                continue
            seen_run = None
            for e in a.events:
                if e["to"] == "RUNNING":
                    seen_run = e["t"]
            if seen_run is None:
                continue
            for d in deps:
                d_acb = self.acbs.get(d)
                if d_acb is None:
                    self.invariants.append(
                        {"invariant": "I1", "task": a.task_id, "ok": False,
                         "detail": f"missing dependency {d}"})
                    continue
                d_done = [e["t"] for e in d_acb.events if e["to"] == "DONE"]
                ok = bool(d_done) and min(d_done) <= seen_run
                self.invariants.append(
                    {"invariant": "I1", "task": a.task_id, "dep": d, "ok": ok})
        # I2: every terminal task has a recorded outcome
        for a in self.acbs.values():
            if a.state in TERMINAL:
                self.invariants.append(
                    {"invariant": "I2", "task": a.task_id, "ok": a.events[-1]["to"] == a.state})
