"""Offline unit checks for src/agent_runtime/acb.py — no GPU, no models.

Covers: state machine transitions, DAG dependency ordering (incl. runtime
growth via repairs), GPU-budget admission accounting, heartbeat timeout ->
FAILED -> repair, and the I1/I2 invariants. Run: python scripts/check_acb.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_runtime.acb import AgentControlBlock, ACBScheduler  # noqa: E402


class FakeBackend:
    def __init__(self, budget=1000, behaviors=None):
        self.gpu_budget_bytes = budget
        self.behaviors = behaviors or {}   # task_id -> "ok" | "reject" | "error" | "hang"
        self.executed = []

    def heartbeat(self, acb):
        return None if self.behaviors.get(acb.task_id) == "hang" else acb.last_heartbeat

    def execute(self, acb):
        self.executed.append(acb.task_id)
        b = self.behaviors.get(acb.task_id, "ok")
        if b == "error":
            return {"artifact": None, "gate": None, "error": "boom",
                    "wait_states": {"WAIT_MODEL": 0.5}}
        if b == "reject":
            return {"artifact": "bad", "gate": {"accepted": False, "reason": "wrong"},
                    "error": None, "wait_states": {"WAIT_MODEL": 0.4}}
        return {"artifact": "good", "gate": {"accepted": True, "reason": "ok"},
                "error": None, "wait_states": {"WAIT_MODEL": 0.3, "WAIT_FUSION": 0.1}}


def acb(tid, **kw):
    return AgentControlBlock(task_id=tid, command=f"cmd-{tid}", **kw)


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        raise SystemExit(f"check failed: {name}")


# 1. linear DAG: b waits for a
be = FakeBackend()
s = ACBScheduler(be)
s.submit(acb("a", gpu_bytes_est=600))
s.submit(acb("b", depends_on=("a",), gpu_bytes_est=600))
rep = s.run()
order = be.executed
check("I1: b runs after a", order.index("b") > order.index("a"))
check("both done", rep["done"] == 2)
check("b waited on dependency",
      rep["tasks"]["b"]["wait_dependency_s"] >= 0.0 and
      any(e["to"] == "WAIT_DEPENDENCY" for e in s.acbs["b"].events))
check("invariants all ok", all(i["ok"] for i in rep["invariants"]))
check("wait_model measured", rep["tasks"]["a"]["wait_model_s"] > 0.2)
check("wait_fusion measured", rep["tasks"]["a"]["wait_fusion_s"] > 0.05)

# 2. GPU budget admission: two fat tasks never exceed budget
be = FakeBackend(budget=1000)
s = ACBScheduler(be)
s.submit(acb("a", gpu_bytes_est=700, priority=2))
s.submit(acb("b", gpu_bytes_est=700, priority=1))
s.submit(acb("c", gpu_bytes_est=300, priority=3))
rep = s.run()
check("all done despite budget", rep["done"] == 3)
check("serial execution under budget", be.executed == ["c", "a", "b"])
emits = [e for e in s.emit.__self__.__dict__.get("_log", [])] if False else None

# 3. gate reject -> repair task -> dynamic growth
be = FakeBackend(behaviors={"a": "reject", "a.r1": "ok"})
log = []
s = ACBScheduler(be, repair_policy=lambda a: f"retry {a.task_id}",
                 emit=log.append)
s.submit(acb("a", gpu_bytes_est=100))
rep = s.run()
check("repair ran", "a.r1" in be.executed)
check("original failed kept", s.acbs["a"].state in ("FAILED", "REPAIR_PENDING"))
check("repair passed", s.acbs["a.r1"].state == "DONE")
check("dynamic growth counted", rep["repair_count"] == 1)
check("repair emitted", any(e["event"] == "repair_submit" for e in log))
check("invariants ok with repair", all(i["ok"] for i in rep["invariants"]))

# 4. attempts exhausted -> no infinite repairs
be = FakeBackend(behaviors={"a": "reject", "a.r1": "reject"})
s = ACBScheduler(be, repair_policy=lambda a: f"retry {a.task_id}")
s.submit(acb("a", gpu_bytes_est=100, max_attempts=2))
rep = s.run()
check("bounded repairs", "a.r2" not in be.executed and rep["done"] == 0)

# 5. heartbeat loss -> FAILED -> repair
be = FakeBackend(behaviors={"h": "hang"})
s = ACBScheduler(be, repair_policy=lambda a: "recover")
s.submit(acb("h", gpu_bytes_est=100))
rep = s.run()
check("heartbeat failure detected", s.acbs["h"].state in ("FAILED", "REPAIR_PENDING"))
check("recovery repair ran", "h.r1" in be.executed and s.acbs["h.r1"].state == "DONE")

# 6. backend error -> FAILED, no repair when policy returns None
be = FakeBackend(behaviors={"x": "error"})
s = ACBScheduler(be, repair_policy=lambda a: None)
s.submit(acb("x", gpu_bytes_est=100))
rep = s.run()
check("error kept in denominator", s.acbs["x"].state == "FAILED" and rep["done"] == 0)

# 7. dependency failure cascades honestly (no fake success)
be = FakeBackend(behaviors={"a": "reject"})
s = ACBScheduler(be, repair_policy=lambda a: None)
s.submit(acb("a", gpu_bytes_est=100))
s.submit(acb("b", depends_on=("a",), gpu_bytes_est=100))
rep = s.run()
check("dependent failed not done", s.acbs["b"].state == "FAILED")
check("no execution of dependent", "b" not in be.executed)

print("\nALL ACB CHECKS PASSED")
