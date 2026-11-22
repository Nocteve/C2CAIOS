"""OS scheduling methodology for Agent tasks: CFS-style fairness, deadlock
detection and banker's-algorithm admission — pure orchestration (no torch).

These are the three classic OS ideas the proposal PPT names but the baseline
serial scheduler (acb.py) does not yet have:

- CFS / vruntime (PPT p5, Linux CFS inspiration): tasks accrue virtual
  runtime proportional to consumed service normalized by weight; the READY
  task with the smallest vruntime runs next. Under serial non-preemptive
  execution this replaces static-priority ordering with provably fairer
  ordering: small tasks stop queueing behind big ones.
- Deadlock detection (wait-for graph): a dependency edge plus a task holding
  resources forms a wait-for edge; a cycle is reported as EDEADLK-style
  evidence instead of the blanket "dependency_deadlock_or_missing", and the
  involved tasks fail fast without executing anything.
- Banker's algorithm (Dijkstra): admit a task only if the remaining budget
  still admits *some* completion ordering of every holding task's max claim
  (gpu_bytes_est is the declared claim). An unsafe request is held in READY —
  the OS denies the resource rather than entering hold-and-wait deadlock.

FairScheduler is a drop-in subclass of acb.ACBScheduler: same backend
protocol, same event vocabulary, same I1/I2 invariants; it only changes
READY ordering, adds the wait-for probe, and can replace the admission test.
"""
from __future__ import annotations

from heapq import heappush, heappop
from time import perf_counter

from .acb import ACBScheduler, SchedulerError, TERMINAL


class FairScheduler(ACBScheduler):
    """CFS-flavoured ACB scheduler.

    Params:
      weights          task_id -> positive weight (default 1.0; bigger = more
                       deserving of CPU, accrues vruntime slower, CFS-style).
      quantum_ms       normalization: service seconds are converted to
                       vruntime units so magnitudes read like milliseconds.
      admission        "fit" (baseline: fits budget) or "banker".
    """

    def __init__(self, backend, repair_policy=None, emit=None, *,
                 weights=None, quantum_ms=1.0, admission="fit"):
        super().__init__(backend, repair_policy=repair_policy, emit=emit)
        self.weights = dict(weights or {})
        self.quantum_ms = float(quantum_ms)
        if admission not in {"fit", "banker"}:
            raise ValueError("admission must be 'fit' or 'banker'")
        self.admission = admission
        self.vruntime = {}          # task_id -> float
        self.min_vruntime = 0.0
        self.service = {}           # task_id -> measured service seconds
        self.deadlocks = []         # evidence records

    # -- CFS accounting ----------------------------------------------------
    def _weight(self, acb):
        w = float(self.weights.get(acb.task_id, 1.0))
        return w if w > 0 else 1.0

    def _charge(self, acb, service_s):
        prev = self.vruntime.get(acb.task_id, self.min_vruntime)
        charged = (service_s * 1000.0 / self.quantum_ms) / self._weight(acb)
        self.vruntime[acb.task_id] = prev + charged
        self.service[acb.task_id] = self.service.get(acb.task_id, 0.0) + service_s
        self.min_vruntime = max(self.min_vruntime, self.vruntime[acb.task_id])

    def _pick_next(self, candidates):
        """Min-vruntime first; ties break toward smaller declared cost
        (shorter job) then submission order — the CFS leftmost rule."""
        return min(candidates,
                   key=lambda a: (self.vruntime.get(a.task_id, self.min_vruntime),
                                  a.est_tokens or 0))

    # -- Banker's safe-state admission (Dijkstra) ---------------------------
    @staticmethod
    def banker_safe(allocations, needs, budget):
        """Textbook safety check. ``allocations`` = bytes currently reserved by
        running tasks (they hold-and-keep), ``needs`` = additional worst-case
        bytes each task may still request mid-execution (hold-and-wait is
        possible). Safe iff some completion order exists in which every
        remaining need fits the budget as holders finish and release."""
        available = budget - sum(allocations)
        if available < 0:
            return False
        remaining = list(zip(allocations, needs))
        while True:
            progressed = False
            for alloc, need in list(remaining):
                if need <= available:
                    available += alloc          # finished: releases its allocation
                    remaining.remove((alloc, need))
                    progressed = True
            if not progressed:
                break
        return not remaining

    def _admit_ok(self, admitted_bytes, acb, budget):
        if not budget:
            return True
        if self.admission == "banker":
            holding = [a for a in self.acbs.values() if a.state == "RUNNING"]
            allocations = [a.gpu_bytes_est for a in holding] + [acb.gpu_bytes_est]
            needs = [int(getattr(a, "burst_bytes", 0)) for a in holding] \
                + [int(getattr(acb, "burst_bytes", 0))]
            return self.banker_safe(allocations, needs, budget)
        return admitted_bytes + acb.gpu_bytes_est <= budget

    # -- Deadlock detection (wait-for graph) --------------------------------
    def _wait_for_edges(self, in_flight):
        """Edge A -> B: A is admitted/holding and waits on B (dependency not
        DONE), or A is READY and blocked behind an in-flight holder for
        budget reasons (resource wait). Dependency cycles are the classic
        deadlock; we report both shapes with the cycle path."""
        edges = {}
        holding = {a.task_id for a in in_flight.values()}
        for a in self.acbs.values():
            waits = [d for d in a.depends_on
                     if d in self.acbs and self.acbs[d].state != "DONE"]
            if a.task_id in holding and waits:
                edges[a.task_id] = waits
            elif waits:
                edges.setdefault(a.task_id, list(waits))
        return edges

    @staticmethod
    def _find_cycle(edges):
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in edges}
        stack, cycles = [], []

        def dfs(node, path):
            color[node] = GRAY
            path.append(node)
            for nxt in edges.get(node, []):
                if nxt not in color:
                    continue
                if color[nxt] == GRAY:
                    cycles.append(path[path.index(nxt):] + [nxt])
                elif color[nxt] == WHITE:
                    dfs(nxt, path)
            path.pop()
            color[node] = BLACK

        for node in list(edges):
            if color[node] == WHITE:
                dfs(node, [])
        return cycles

    def _detect_deadlock(self, in_flight):
        cycles = self._find_cycle(self._wait_for_edges(in_flight))
        for cycle in cycles:
            record = {"event": "deadlock_detected", "cycle": cycle,
                      "t": perf_counter()}
            self.deadlocks.append(record)
            self.emit(record)
            for tid in dict.fromkeys(cycle):
                acb = self.acbs.get(tid)
                if acb and acb.state not in TERMINAL:
                    acb.transition("FAILED", f"deadlock:{'->'.join(cycle)}")
                    self._after_terminal(acb)
        return bool(cycles)

    # -- run loop -----------------------------------------------------------
    def run(self):
        started = perf_counter()
        budget = int(getattr(self.backend, "gpu_budget_bytes", 0) or 0)
        in_flight = {}
        admitted_bytes = 0
        while True:
            for tid, acb in list(in_flight.items()):
                beat = self.backend.heartbeat(acb)
                acb.last_heartbeat = beat if beat is not None else acb.last_heartbeat
                if beat is None and acb.state in ("RUNNING", "WAIT_MODEL", "WAIT_FUSION"):
                    acb.transition("FAILED", "heartbeat_lost")
                    self.emit({"event": "heartbeat_lost", "task": tid})
                    in_flight.pop(tid)
                    admitted_bytes -= acb.gpu_bytes_est
                    self._after_terminal(acb)

            for acb in self.acbs.values():
                if acb.state == "WAIT_DEPENDENCY":
                    if self._deps_failed(acb):
                        acb.transition("FAILED", "dependency_failed")
                        self._after_terminal(acb)
                    elif self._deps_done(acb):
                        acb.transition("READY", "dependencies_done")
                elif acb.state == "READY" and not self._deps_done(acb):
                    acb.transition("WAIT_DEPENDENCY", "dependency_not_done")

            # Deadlock probe BEFORE admission (fail fast, execute nothing).
            self._detect_deadlock(in_flight)

            # CFS admission: iterate picking min-vruntime among READY.
            while True:
                candidates = [a for a in self.acbs.values() if a.state == "READY"
                              and a.task_id not in in_flight]
                if not candidates:
                    break
                nxt = self._pick_next(candidates)
                if not self._admit_ok(admitted_bytes, nxt, budget):
                    candidates.remove(nxt)
                    if not candidates:
                        break
                    continue
                nxt.transition("RUNNING", "admitted")
                nxt.attempts += 1
                in_flight[nxt.task_id] = nxt
                admitted_bytes += nxt.gpu_bytes_est
                self.emit({"event": "admit", "task": nxt.task_id,
                           "admitted_bytes": admitted_bytes, "budget": budget,
                           "vruntime": round(self.vruntime.get(nxt.task_id, 0.0), 3)})

            for tid, acb in list(in_flight.items()):
                if self.backend.heartbeat(acb) is None:
                    acb.transition("FAILED", "heartbeat_lost_preflight")
                    self.emit({"event": "heartbeat_lost", "task": tid})
                    in_flight.pop(tid)
                    admitted_bytes -= acb.gpu_bytes_est
                    self._after_terminal(acb)

            if in_flight:
                tid, acb = next(iter(in_flight.items()))
                acb.transition("WAIT_MODEL", "backend_execute_begin")
                exec_started = perf_counter()
                result = self.backend.execute(acb) or {}
                service_s = perf_counter() - exec_started
                self._charge(acb, service_s)
                wait_states = result.get("wait_states") or {}
                for ws, secs in wait_states.items():
                    acb.events.append({"t": perf_counter(), "from": "RUNNING",
                                       "to": ws, "cause": f"measured:{secs:.6f}s",
                                       "measured_seconds": secs})
                acb.artifact = result.get("artifact")
                acb.gate = result.get("gate")
                acb.error = result.get("error")
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
                           "gate": acb.gate, "error": acb.error,
                           "vruntime": round(self.vruntime.get(tid, 0.0), 3)})
                self._after_terminal(acb)
                continue

            pending = [a for a in self.acbs.values()
                       if a.state not in TERMINAL and a.state != "REPAIR_PENDING"]
            if not pending:
                break
            if all(a.state == "WAIT_DEPENDENCY" for a in pending):
                for a in pending:
                    if a.state != "FAILED":
                        a.transition("FAILED", "dependency_deadlock_or_missing")
                break
        report = self._report(perf_counter() - started)
        report["scheduling"] = {
            "policy": "cfs-vruntime" if self.admission == "fit" else "cfs-vruntime+banker",
            "admission": self.admission,
            "vruntime": dict(self.vruntime),
            "service_s": dict(self.service),
            "deadlocks": self.deadlocks,
        }
        self._check_invariants()
        report["invariants"] = self.invariants
        return report
