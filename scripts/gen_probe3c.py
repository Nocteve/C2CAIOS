"""Conditional-decision variants of the probe3 family (probe3c, 2026-10-17).

Lever C: probe3's pending value only blocked a controller, so the Planner
could delegate without computing it (analysis completeness 15/30 in probe3f).
probe3c makes the value the condition of a plan branch the Planner itself must
choose and state in its action reason (multi3 design, validated 2026-10-16),
so computing the value is a prerequisite of the Planner's own output.

Everything else is byte-identical to the base probe3 scenario: same tasks,
events, current_state, probe (question and expected answer). Only the goal and
the controller note change. Thresholds are generic per-field constants and are
asserted never to equal an expected answer.
"""
import json
import re
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "c2c-core-os"

# field -> (condition text, plan if true, plan if false)
RULES = {
    "remaining_memory_after_ready": ("stays at or above 8192 MB", "admit every READY task", "defer the smallest READY task"),
    "remaining_memory_after_release": ("stays at or above 8192 MB", "admit the waiting successor now", "defer the waiting successor"),
    "remaining_memory_after_handoff": ("stays at or above 8192 MB", "admit the next stage now", "defer the next stage"),
    "remaining_memory_after_replicas": ("stays at or above 8192 MB", "admit all replicas now", "defer one replica"),
    "remaining_memory_after_reserve": ("stays at or above 8192 MB", "admit every READY job", "defer the smallest READY job"),
    "effective_timeout_margin_seconds": ("stays at or above 30 s", "let the task keep running", "restart the task now"),
    "start_slack_seconds": ("stays at or least 60 s".replace("at or least", "at least"), "admit the deadline task this cycle", "start the deadline task immediately with priority"),
    "worst_case_backlog_seconds": ("exceeds 300 s", "escalate to the incident team", "let the remaining retries finish"),
    "queue_wait_seconds": ("exceeds 300 s", "notify the submitter of the wait", "let the queue drain in order"),
    "gpu_waves_required": ("is at most 2", "reserve a single-wave GPU plan", "reserve a multi-wave GPU plan"),
    "initial_transfer_seconds": ("fits within 120 s", "start the pipeline now", "delay the pipeline start"),
    "critical_path_seconds": ("fits within 120 s", "commit the resource plan now", "revise the resource plan before committing"),
    "batch_makespan_seconds": ("fits within 120 s", "launch the batch now", "delay the batch launch"),
    "free_cpu_after_turnover": ("stays at or above 4 slots", "admit the waiting job now", "defer the waiting job"),
    "free_cpu_after_eviction_and_admission": ("stays at or above 4 slots", "admit the replacement job now", "defer the replacement job"),
}


def standalone_integers(obj):
    text = json.dumps(obj, ensure_ascii=False)
    return {int(m) for m in re.findall(r"(?<![\d.])(\d+)(?![\d.])", text)}


def main():
    written = 0
    for path in sorted(OUT.glob("public-probe3-*.json")):
        base = json.loads(path.read_text(encoding="utf-8"))
        slug = base["id"].split("probe3-")[1]
        pending = base["public_background"]["pending_decision"]
        field = pending["field"]
        rule = RULES[field]
        cond, plan_t, plan_f = rule
        threshold_probe = {"remaining_memory": 8192, "seconds": None}
        goal = (
            f"Choose the next scheduling plan by this rule: if {field} {cond}, "
            f"the plan is: {plan_t}; otherwise: {plan_f}. Your action reason "
            "must state the chosen plan together with the value that determines "
            "it, and you must delegate one bounded verification task for that "
            "value. Preserve dependencies, resource limits and failure evidence."
        )
        case = json.loads(json.dumps(base))
        case["id"] = base["id"].replace("probe3-", "probe3c-")
        case["goal"] = goal
        case["public_background"]["pending_decision"]["controller_note"] = (
            f"The plan branch for {field} depends on this value; it must be "
            "verified before the controller executes the chosen plan.")
        answer = case["probe"]["expected_answer"]
        public_ints = standalone_integers(
            {"goal": case["goal"], "public_background": case["public_background"],
             "current_state": case["current_state"]})
        assert answer not in public_ints, f"{case['id']}: answer {answer} leaks"
        # threshold must not equal the answer (generic constants chosen far from answers)
        fname = f"public-probe3c-{slug}.json"
        (OUT / fname).write_text(json.dumps(case, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
        print(f"wrote {fname}: field={field} answer={answer}")
        written += 1
    print(f"total {written} probe3c scenarios")


if __name__ == "__main__":
    main()
