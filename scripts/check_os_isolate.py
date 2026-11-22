"""Self-test for os_isolate: real containment evidence for the PPT fault-isolation design.

Runs only OS-level probes (no GPU / no torch). Expected to pass fully on a
systemd user manager (WSL2 with systemd); the plain-subprocess backend is
exercised explicitly so the fallback is evidence-backed too.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent_runtime.os_isolate import (IsolationLimits, run_isolated,
                                      systemd_scope_available)

PY = sys.executable
results = []


def check(name, ok, detail):
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} :: {detail}")


def units_left():
    out = subprocess.run(["systemctl", "--user", "list-units", "--all", "--plain",
                          "--no-legend", "--no-pager"], capture_output=True, text=True)
    return [line.split()[0] for line in out.stdout.splitlines()
            if line.startswith("c2cos-agent")]


scope_ok = systemd_scope_available()
check("backend-detection", True,
      f"systemd-scope available: {scope_ok}")

# 1. Trivial command under isolation
r = run_isolated([PY, "-c", "print('ok')"], IsolationLimits(timeout_s=30))
check("trivial-exit0", r.rc == 0 and not r.oom_killed and not r.timed_out,
      f"rc={r.rc} isolation={r.isolation} wall={r.wall_s:.2f}s")

# 2. Memory bomb contained (150 MiB allocations under a 64 MiB cap)
bomb = ("a=[]\n"
        "for i in range(6):\n"
        "    a.append(bytearray(25*1024*1024))\n"
        "    print('alloc', i, flush=True)\n"
        "print('bomb-survived', flush=True)")
r = run_isolated([PY, "-c", bomb], IsolationLimits(memory_mib=64, timeout_s=60))
parent_alive = True  # reaching this line already proves the parent survived
enforcement = (r.extra or {}).get("memory_enforcement")
oom_detected = r.oom_killed or (r.rc not in (0, None) and r.rc < 0)
check("memory-bomb-contained",
      parent_alive and not r.timed_out and r.rc != 0,
      f"rc={r.rc} oom_killed={r.oom_killed} isolation={r.isolation} "
      f"enforcement={enforcement} peak={r.memory_peak_mib and round(r.memory_peak_mib, 1)}MiB "
      f"wall={r.wall_s:.2f}s")
if r.isolation == "systemd-scope" and enforcement == "cgroup-memory.max":
    check("memory-bomb-oom-flag", r.oom_killed,
          f"unit Result reports oom-kill: {r.oom_killed}")
else:
    check("memory-bomb-oom-flag", True,
          f"enforcement={enforcement} (RLIMIT_AS raises MemoryError in-child; "
          "cgroup oom-kill flag not applicable)")

# 3. Fork bomb contained by TasksMax
forkbomb = ("import os\n"
            "while True:\n"
            "    os.fork()")
r = run_isolated([PY, "-c", forkbomb],
                 IsolationLimits(pids_max=32, memory_mib=256, timeout_s=60))
check("fork-bomb-contained", r.rc != 0 and r.wall_s < 60,
      f"rc={r.rc} wall={r.wall_s:.2f}s isolation={r.isolation}")

# 4. Wall-clock leash
r = run_isolated([PY, "-c", "import time; time.sleep(120)"],
                 IsolationLimits(timeout_s=2))
check("timeout-leash", r.timed_out and r.rc != 0 and r.wall_s < 30,
      f"timed_out={r.timed_out} wall={r.wall_s:.2f}s")

# 5. Crash does not propagate (child segfault-style -9 self-kill)
r = run_isolated([PY, "-c", "import os,signal; os.kill(os.getpid(), signal.SIGKILL)"],
                 IsolationLimits(timeout_s=30))
check("child-crash-contained", r.rc != 0,
      f"rc={r.rc} parent-alive=True")

# 6. Plain fallback backend also works
r = run_isolated([PY, "-c", "print('ok')"], IsolationLimits(timeout_s=30),
                 prefer_systemd=False)
check("plain-fallback", r.rc == 0 and r.isolation == "plain-subprocess",
      f"rc={r.rc} isolation={r.isolation}")

# 7. No leftover scope units / processes
time.sleep(1)
leftover = units_left()
check("cleanup", not leftover, f"leftover c2cos-agent units: {leftover}")

summary = {
    "systemd_scope_available": scope_ok,
    "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in results],
    "all_passed": all(ok for _, ok, _ in results),
}
Path("artifacts/os-isolate-selftest.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
print("\nALL PASS" if summary["all_passed"] else "\nSOME FAILED")
sys.exit(0 if summary["all_passed"] else 1)
