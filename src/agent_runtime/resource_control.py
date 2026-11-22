"""Simulated cgroup v2 resource controller for Agent tasks.

Where the kernel's cgroup tree is unavailable (read-only /sys/fs/cgroup in
containers, WSL without controller delegation), the runtime simulates the
cgroup v2 *interface and semantics* in userspace — the same methodology early
container tooling used before cgroup v1/v2 matured:

- A per-task directory tree mirrors cgroup v2 files: memory.max, memory.current,
  memory.peak, pids.max, pids.current, cpu.max, memory.events (oom_kill),
  cpu.stat (throttled). Values are real telemetry where the backend measures
  them (decode tokens, incremental GPU MiB, service seconds).
- Enforcement points mirror the kernel:
  * admission: slice pids.current == pids.max -> the request is denied
    (cgroup fork-fail semantics) — "resource temporarily unavailable";
  * metering: usage is charged against quotas as it is reported;
  * violation: exceeding a hard quota kills the task into FAILED with an
    oom_kill-style event, exactly like the kernel OOM killer — the runtime
    survives and the scheduler's bounded-repair path engages.
- Everything is written to disk as it happens, so the simulated cgroup
  filesystem is inspectable during and after a run (cat tasks/t3/memory.current).

Honest limits: charging is event-driven from backend telemetry, not sampled
every jiffy like the kernel; cpu.max throttling is recorded, not preempted
(there is no userspace clock tick without a watchdog thread). Reports must
say "simulated" next to every number.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TaskQuota:
    tokens: int | None = None        # max decode tokens (like memory.max)
    wall_s: float | None = None      # max service seconds
    memory_mib: float | None = None  # max incremental GPU memory
    pids: int | None = None          # max child processes (TasksMax analog)


@dataclass
class SliceQuota:
    pids_max: int | None = None          # total live tasks (cgroup pids.max)
    concurrency_max: int | None = None   # max concurrently RUNNING tasks


class SimulatedCgroupFs:
    """Live cgroup v2 file tree under <root>/<slice>/<task>/."""

    def __init__(self, root, slice_name="agent.slice"):
        self.root = Path(root)
        self.slice = self.root / slice_name
        self.slice.mkdir(parents=True, exist_ok=True)
        self._write_slice_defaults()

    def _write_slice_defaults(self):
        (self.slice / "cgroup.controllers").write_text("cpu memory pids\n")
        (self.slice / "cgroup.type").write_text("domain\n")

    def task_dir(self, task_id):
        d = self.slice / f"task-{task_id}.scope"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def publish(self, task_id, quota: TaskQuota):
        d = self.task_dir(task_id)
        (d / "memory.max").write_text(
            "max" if quota.memory_mib is None else str(int(quota.memory_mib * 2 ** 20)))
        (d / "pids.max").write_text(
            "max" if quota.pids is None else str(int(quota.pids)))
        (d / "memory.current").write_text("0")
        (d / "memory.peak").write_text("0")
        (d / "pids.current").write_text("0")
        (d / "cpu.max").write_text("max 100000")
        (d / "memory.events").write_text("low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n")
        (d / "cpu.stat").write_text("usage_usec 0\nnr_periods 0\nnr_throttled 0\nthrottled_usec 0\n")
        (d / "tokens.current").write_text("0")
        return d

    def charge(self, task_id, *, memory_mib=0.0, tokens=0, pids=0,
               wall_s=0.0, throttle_s=0.0):
        """Add usage; returns (violation_kind, detail) or (None, None)."""
        d = self.task_dir(task_id)
        mem = float((d / "memory.current").read_text()) + memory_mib * 2 ** 20
        peak = max(float((d / "memory.peak").read_text()), mem)
        (d / "memory.current").write_text(str(int(mem)))
        (d / "memory.peak").write_text(str(int(peak)))
        tok = int((d / "tokens.current").read_text()) + int(tokens)
        (d / "tokens.current").write_text(str(tok))
        pid = int((d / "pids.current").read_text()) + int(pids)
        (d / "pids.current").write_text(str(pid))
        stat = (d / "cpu.stat").read_text().splitlines()
        usage = int(stat[0].split()[1]) + int(wall_s * 1e6)
        throttled = int(stat[3].split()[1]) + int(throttle_s * 1e6)
        (d / "cpu.stat").write_text(
            f"usage_usec {usage}\nnr_periods 0\nnr_throttled 0\nthrottled_usec {throttled}\n")
        mem_max = (d / "memory.max").read_text().strip()
        if mem_max != "max" and mem > float(mem_max):
            self._oom_event(d)
            return "memory_mib", f"{mem / 2 ** 20:.1f}>{float(mem_max) / 2 ** 20:.0f}"
        if quota_has((d / "quota.json"), "tokens") and tok > quota_get((d / "quota.json"), "tokens"):
            return "tokens", f"{tok}>{quota_get((d / 'quota.json'), 'tokens')}"
        if quota_has((d / "quota.json"), "wall_s") and (usage / 1e6) > quota_get((d / "quota.json"), "wall_s"):
            return "wall_s", f"{usage / 1e6:.1f}>{quota_get((d / 'quota.json'), 'wall_s')}"
        if quota_has((d / "quota.json"), "pids") and pid > quota_get((d / "quota.json"), "pids"):
            return "pids", f"{pid}>{quota_get((d / 'quota.json'), 'pids')}"
        return None, None

    def _oom_event(self, d):
        events = (d / "memory.events").read_text().splitlines()
        out = []
        for line in events:
            k, v = line.split()
            if k == "oom_kill":
                v = str(int(v) + 1)
            if k == "oom":
                v = str(int(v) + 1)
            out.append(f"{k} {v}")
        (d / "memory.events").write_text("\n".join(out) + "\n")

    def set_quota(self, task_id, quota: TaskQuota):
        d = self.task_dir(task_id)
        (d / "quota.json").write_text(json.dumps({
            k: v for k, v in vars(quota).items() if v is not None}))


def quota_has(path, key):
    try:
        return key in json.loads(Path(path).read_text())
    except Exception:
        return False


def quota_get(path, key):
    return json.loads(Path(path).read_text())[key]


class ResourceController:
    """Slice-level accounting + enforcement, wrapping a scheduler backend."""

    def __init__(self, cgroupfs: SimulatedCgroupFs, task_quota: TaskQuota = None,
                 slice_quota: SliceQuota = None, emit=None):
        self.fs = cgroupfs
        self.task_quota = task_quota or TaskQuota()
        self.slice_quota = slice_quota or SliceQuota()
        self.emit = emit or (lambda rec: None)
        self.live = set()
        self.violations = []

    def _slice_files(self):
        d = self.fs.slice
        pids_cur = len(self.live)
        (d / "pids.current").write_text(str(pids_cur))
        (d / "pids.max").write_text(
            "max" if self.slice_quota.pids_max is None else str(self.slice_quota.pids_max))
        conc = getattr(self, "_running", 0)
        (d / "concurrency.current").write_text(str(conc))
        (d / "concurrency.max").write_text(
            "max" if self.slice_quota.concurrency_max is None
            else str(self.slice_quota.concurrency_max))

    def try_admit(self, acb) -> tuple[bool, str | None]:
        """cgroup pids.max fork-fail semantics at admission."""
        self._slice_files()
        if self.slice_quota.pids_max is not None and len(self.live) >= self.slice_quota.pids_max:
            return False, f"slice pids.max {self.slice_quota.pids_max} reached (fork fail)"
        self.live.add(acb.task_id)
        self.fs.publish(acb.task_id, self.task_quota)
        self.fs.set_quota(acb.task_id, self.task_quota)
        return True, None

    def release(self, acb):
        self.live.discard(acb.task_id)
        self._slice_files()

    def charge(self, acb, *, memory_mib=0.0, tokens=0, pids=0, wall_s=0.0,
               throttle_s=0.0) -> tuple[bool, str | None]:
        kind, detail = self.fs.charge(
            acb.task_id, memory_mib=memory_mib, tokens=tokens, pids=pids,
            wall_s=wall_s, throttle_s=throttle_s)
        if kind:
            rec = {"event": "quota_violation", "task": acb.task_id,
                   "kind": kind, "detail": detail, "t": time.perf_counter()}
            self.violations.append(rec)
            self.emit(rec)
            return False, f"{kind} quota exceeded: {detail}"
        return True, None

    def report(self):
        usage = {}
        for scope in sorted(self.fs.slice.glob("task-*.scope")):
            def rd(name, default="0"):
                p = scope / name
                return p.read_text().strip() if p.exists() else default
            usage[scope.name] = {
                "memory_current_mib": round(float(rd("memory.current", "0")) / 2 ** 20, 2),
                "memory_peak_mib": round(float(rd("memory.peak", "0")) / 2 ** 20, 2),
                "tokens": int(rd("tokens.current")),
                "pids": int(rd("pids.current")),
                "usage_s": round(int(rd("cpu.stat", "usage_usec 0").splitlines()[0].split()[1]) / 1e6, 3),
                "oom_kill": int([l for l in rd("memory.events").splitlines() if l.startswith("oom_kill")][0].split()[1]),
            }
        return {"simulated": True, "slice": self.fs.slice.name,
                "tasks": usage, "violations": self.violations}


class ControlledBackend:
    """Backend decorator: meters real execute() results into the controller
    and turns hard quota violations into contained FAILED outcomes."""

    def __init__(self, inner, controller: ResourceController):
        self.inner = inner
        self.controller = controller
        self.gpu_budget_bytes = getattr(inner, "gpu_budget_bytes", 0)

    def heartbeat(self, acb):
        return self.inner.heartbeat(acb)

    def execute(self, acb):
        self.controller._running = getattr(self.controller, "_running", 0) + 1
        try:
            ok, why = self.controller.try_admit(acb)
            if not ok:
                return {"artifact": None, "gate": None, "error": why,
                        "wait_states": {"WAIT_MODEL": 0.0}}
            result = self.inner.execute(acb) or {}
            tokens, mem = 0, 0.0
            record = result.get("record")  # raw arm record when the backend provides it
            if isinstance(record, dict):
                tokens = int(record.get("output_tokens") or 0)
                mem = float(record.get("peak_incremental_mib") or 0.0)
            wall = 0.0
            for secs in (result.get("wait_states") or {}).values():
                wall += float(secs)
            ok, why = self.controller.charge(
                acb, memory_mib=mem, tokens=tokens, wall_s=wall)
            if not ok:
                # kernel-OOM-killer semantics: the task is killed into FAILED
                # regardless of what the gate said; the runtime survives.
                return {"artifact": None, "gate": None,
                        "error": f"resource_quota_exceeded:{why}",
                        "wait_states": result.get("wait_states") or {}}
            return result
        finally:
            self.controller.release(acb)
            self.controller._running = getattr(self.controller, "_running", 0) - 1
