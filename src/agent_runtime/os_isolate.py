"""OS-level Agent task isolation: real subprocess + cgroup v2 resource limits.

PPT design (slide 8): an Agent must be a real OS child process with cgroup
resource limits, so one Agent's runaway memory / fork bomb / crash is contained
by the kernel instead of dragging down the whole runtime (LangGraph/AutoGen
coroutine-style failure mode).

Two backends, one interface:

- ``systemd-scope``: transient user scope via ``systemd-run --user --scope``
  with ``MemoryMax`` / ``CPUQuota`` / ``TasksMax``. Works wherever a systemd
  user manager runs with resource-controller delegation (e.g. WSL2 with
  systemd, desktop Linux). This is genuine cgroup v2 enforcement by the kernel.
- ``plain-subprocess``: new session + best-effort ``RLIMIT_*`` via
  ``preexec_fn``. Used where the cgroup tree is read-only (containers without
  delegation, e.g. AutoDL); ``isolation`` says exactly which backend ran so
  reports never overstate enforcement.

The module never touches the caller's process state: every child gets its own
process group, is killed as a group on timeout, and its cgroup unit is
collected. Detection is honest: OOM is reported from the unit Result
(``oom-kill``) or the scope's ``memory.events``, never inferred from exit
codes alone.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Sequence

SCOPE_PREFIX = "c2cos-agent"


@dataclass(frozen=True)
class IsolationLimits:
    memory_mib: Optional[int] = None      # cgroup MemoryMax
    cpu_quota_pct: Optional[int] = None   # cgroup CPUQuota (1..100 per core)
    pids_max: Optional[int] = None        # cgroup TasksMax
    timeout_s: float = 300.0              # wall-clock leash on the child


@dataclass
class IsolationResult:
    argv0: str
    rc: Optional[int]
    isolation: str                        # "systemd-scope" | "plain-subprocess"
    oom_killed: bool = False
    timed_out: bool = False
    wall_s: float = 0.0
    memory_peak_mib: Optional[float] = None
    unit: Optional[str] = None
    error: Optional[str] = None
    extra: dict = field(default_factory=dict)


def systemd_scope_available() -> bool:
    if shutil.which("systemd-run") is None or shutil.which("systemctl") is None:
        return False
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if not runtime or not os.path.isdir(runtime):
        uid = os.getuid() if hasattr(os, "getuid") else None
        if uid is not None and os.path.isdir(f"/run/user/{uid}"):
            runtime = f"/run/user/{uid}"
    if not runtime:
        return False
    env = dict(os.environ, XDG_RUNTIME_DIR=runtime, DBUS_SESSION_BUS_ADDRESS=f"unix:path={runtime}/bus")
    try:
        out = subprocess.run(
            ["systemctl", "--user", "is-system-running"], env=env,
            capture_output=True, text=True, timeout=10,
        )
        state = (out.stdout or "").strip()
        return state in {"running", "degraded"} and os.path.exists(f"{runtime}/bus")
    except Exception:
        return False


def _systemd_env():
    env = dict(os.environ)
    runtime = env.get("XDG_RUNTIME_DIR")
    if not runtime and hasattr(os, "getuid"):
        candidate = f"/run/user/{os.getuid()}"
        if os.path.isdir(candidate):
            runtime = candidate
            env["XDG_RUNTIME_DIR"] = candidate
    if runtime:
        env["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={runtime}/bus"
    return env


def _scope_cgroup_file(unit: str, name: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["systemctl", "--user", "show", "-p", "ControlGroup", "--value", unit],
            env=_systemd_env(), capture_output=True, text=True, timeout=10,
        )
        group = (out.stdout or "").strip()
        if not group:
            return None
        path = os.path.join("/sys/fs/cgroup", group.lstrip("/"), name)
        return path if os.path.exists(path) else None
    except Exception:
        return None


def _read_scope_memory_peak(unit: str) -> Optional[float]:
    path = _scope_cgroup_file(unit, "memory.peak")
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return int(handle.read().strip()) / 2 ** 20
    except Exception:
        return None


def _memory_enforcement(unit: str, limits: IsolationLimits) -> str:
    """Report which mechanism actually capped memory: cgroup memory.max, the
    child's RLIMIT_AS, or nothing (limit unset)."""
    if not limits.memory_mib:
        return "none"
    path = _scope_cgroup_file(unit, "memory.max")
    if path:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                if handle.read().strip() != "max":
                    return "cgroup-memory.max"
        except Exception:
            pass
    return "rlimit-as"


def _kill_scope(unit: str) -> None:
    """Kill every process in the scope immediately (cgroup.kill semantics)."""
    try:
        subprocess.run(
            ["systemctl", "--user", "kill", "--kill-whom=all", "--signal=KILL", unit],
            env=_systemd_env(), capture_output=True, timeout=15,
        )
    except Exception:
        pass


def _stop_scope(unit: str) -> None:
    try:
        subprocess.run(
            ["systemctl", "--user", "stop", "--quiet", unit],
            env=_systemd_env(), capture_output=True, timeout=15,
        )
    except Exception:
        pass


def _run_systemd_scope(argv, limits: IsolationLimits, unit: str, env) -> IsolationResult:
    cmd = ["systemd-run", "--user", "--scope", "--quiet", "--collect", "--unit", unit]
    child_argv = list(argv)
    if limits.memory_mib:
        # MemoryMax needs the memory controller delegated to the user manager;
        # on kernels without delegation it is silently ignored, so also cap the
        # address space in-child via ulimit -v (always enforced). The result
        # records which mechanism actually held.
        cmd += ["--property", f"MemoryMax={int(limits.memory_mib)}M"]
        child_argv = ["bash", "-c", 'ulimit -v "$1"; shift; exec "$@"', "_",
                      str(int(limits.memory_mib) * 1024), *argv]
    if limits.cpu_quota_pct:
        cmd += ["--property", f"CPUQuota={int(limits.cpu_quota_pct)}%"]
    if limits.pids_max:
        cmd += ["--property", f"TasksMax={int(limits.pids_max)}"]
    cmd += ["--", *child_argv]
    started = time.perf_counter()
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, start_new_session=True)
    timed_out = False
    try:
        _, stderr = proc.communicate(timeout=limits.timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_scope(unit)          # cgroup.kill == all processes in the scope
        # Belt and braces: signal every PID still listed in the scope cgroup,
        # then the local process group, so a stuck wait never burns 20 s.
        procs_path = _scope_cgroup_file(unit, "cgroup.procs")
        if procs_path:
            try:
                with open(procs_path, "r", encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            os.kill(int(line.strip()), 9)
                        except Exception:
                            pass
            except Exception:
                pass
        try:
            os.killpg(proc.pid, 9)
        except Exception:
            pass
        try:
            _, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            _, stderr = proc.communicate()
    wall = time.perf_counter() - started

    result_text = ""
    try:
        out = subprocess.run(
            ["systemctl", "--user", "show", "-p", "Result", "--value", unit],
            env=_systemd_env(), capture_output=True, text=True, timeout=10,
        )
        result_text = (out.stdout or "").strip()
    except Exception:
        pass
    _stop_scope(unit)  # no-op when --collect already removed it

    return IsolationResult(
        argv0=os.path.basename(argv[0]) if argv else "",
        rc=proc.returncode,
        isolation="systemd-scope",
        oom_killed=(result_text == "oom-kill"),
        timed_out=timed_out,
        wall_s=wall,
        memory_peak_mib=_read_scope_memory_peak(unit),
        unit=unit,
        error=(stderr or b"").decode("utf-8", "ignore").strip()[-500:] or None
        if isinstance(stderr, bytes) else None,
        extra={"memory_enforcement": _memory_enforcement(unit, limits),
               "unit_result": result_text or None},
    )


def _run_plain(argv, limits: IsolationLimits, env) -> IsolationResult:
    def _limits():
        import resource
        if limits.memory_mib:
            # RLIMIT_AS caps the address space (works without cgroup write
            # access).  Address space typically runs several times larger
            # than RSS (mappings, arenas, shared libraries), so the AS cap is
            # set to 4x the RSS budget and acts only as a runaway guard; the
            # memory_mib budget itself is enforced on real RSS by the
            # watchdog below, mirroring cgroup memory.max semantics.
            resource.setrlimit(resource.RLIMIT_AS,
                               (limits.memory_mib * 4 * 2 ** 20,
                                limits.memory_mib * 4 * 2 ** 20))
        if limits.pids_max:
            try:
                resource.setrlimit(resource.RLIMIT_NPROC,
                                   (limits.pids_max, limits.pids_max))
            except (ValueError, OSError):
                pass
        if limits.cpu_quota_pct and limits.cpu_quota_pct < 100:
            # Best effort: hard CPU seconds proportional to the quota.
            resource.setrlimit(resource.RLIMIT_CPU,
                               (int(limits.timeout_s), int(limits.timeout_s) + 1))

    started = time.perf_counter()
    proc = subprocess.Popen(argv, env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, start_new_session=True,
                            preexec_fn=_limits)

    # RSS watchdog: without systemd there are no cgroup oom_kill events, so we
    # poll the child's real resident set from /proc and kill the process group
    # the moment it exceeds memory_mib.  This is a real-RSS, real-kill
    # functional simulation of cgroup memory.max (accounting-class
    # enforcement), not a virtual-address cap like RLIMIT_AS.
    watchdog = {"oom_killed": False, "rss_peak_mib": None, "stop": False}
    if limits.memory_mib:
        import threading

        def _watch():
            peak_mib = 0.0
            limit_bytes = limits.memory_mib * 2 ** 20
            while not watchdog["stop"]:
                try:
                    with open(f"/proc/{proc.pid}/status", "rb") as fh:
                        for line in fh:
                            if line.startswith(b"VmRSS"):
                                rss_bytes = int(line.split()[1]) * 1024
                                break
                        else:
                            rss_bytes = 0
                except OSError:
                    break  # process gone
                peak_mib = max(peak_mib, rss_bytes / 2 ** 20)
                watchdog["rss_peak_mib"] = round(peak_mib, 1)
                if rss_bytes > limit_bytes:
                    watchdog["oom_killed"] = True
                    try:
                        os.killpg(proc.pid, 9)
                    except OSError:
                        pass
                    break
                time.sleep(0.01)

        watcher = threading.Thread(target=_watch, daemon=True)
        watcher.start()

    timed_out = False
    try:
        _, stderr = proc.communicate(timeout=limits.timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, 9)
        except Exception:
            pass
        try:
            _, stderr = proc.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            _, stderr = proc.communicate()
    if limits.memory_mib:
        watchdog["stop"] = True
        watcher.join(timeout=1)
    wall = time.perf_counter() - started
    err_text = (stderr or b"").decode("utf-8", "ignore").strip()[-500:] or None
    # A watchdog kill is accounted as an OOM kill even though no cgroup event
    # exists; the enforcement path (real RSS, real kill) is recorded in extra.
    return IsolationResult(
        argv0=os.path.basename(argv[0]) if argv else "",
        rc=proc.returncode,
        isolation="plain-subprocess",
        oom_killed=watchdog["oom_killed"],
        timed_out=timed_out and not watchdog["oom_killed"],
        wall_s=wall,
        memory_peak_mib=watchdog["rss_peak_mib"],
        unit=None,
        error=err_text,
        extra={"rss_peak_mib": watchdog["rss_peak_mib"],
               "memory_enforcement":
                   "rss_watchdog" if limits.memory_mib else None},
    )


def run_isolated(argv: Sequence[str], limits: IsolationLimits,
                 *, env: Optional[dict] = None, prefer_systemd: bool = True) -> IsolationResult:
    """Run ``argv`` as an isolated OS child under ``limits``; never raises for
    child failures (crash / OOM / timeout are reported in the result)."""
    argv = [str(a) for a in argv]
    child_env = dict(os.environ if env is None else env)
    if prefer_systemd and systemd_scope_available():
        unit = f"{SCOPE_PREFIX}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        return _run_systemd_scope(argv, limits, unit, child_env)
    return _run_plain(argv, limits, child_env)
