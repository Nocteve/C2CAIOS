"""Conditional-decision multi-task scenarios (multi3, 2026-10-16).

v6 lesson: dispatch-gate framing (values block a controller decision) still let
the Planner delegate without computing -- its own action (delegation) did not
need the values.  multi3 makes each value the condition of a plan BRANCH the
Planner itself must choose and state in its action reason, so computing the
value is a prerequisite of the Planner's own output.  Thresholds are chosen on
the correct side of every answer and never equal an answer (asserted).

Same subsystems and answers as gen_probe_multi.py; only goal and controller
notes change.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_probe_multi import (  # noqa: E402
    SCENARIOS, standalone_integers)
from gen_probe_multi2 import GATES  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "c2c-core-os"

# scenario-key -> field -> (condition, plan_if_true, plan_if_false)
RULES = {
    "octo-01": {
        "alpha_wait_seconds": ("exceeds 300 s", "preempt qa3 and start alpha_preview early", "notify the submitter of the wait"),
        "beta_wait_seconds": ("exceeds 150 s", "preempt qb1 and start beta_preview early", "notify the submitter of the wait"),
        "pool_remaining_memory_mb": ("stays at or above 20480 MB after admitting all three READY pool jobs", "admit all three pool jobs", "defer the smallest pool job"),
        "gpu_waves_required": ("is at most 2", "reserve a single-wave GPU plan", "reserve a multi-wave GPU plan"),
        "effective_timeout_margin_seconds": ("stays at or above 30 s", "let t1 keep running", "restart t1 now"),
        "worst_case_backlog_seconds": ("exceeds 300 s", "escalate r1 to the incident team", "let the remaining retries finish"),
        "initial_transfer_seconds": ("fits within 600 s", "start the pipeline now", "delay the pipeline start"),
        "start_slack_seconds": ("is at least 120 s", "admit d1 this cycle", "defer d1 to the next cycle"),
    },
    "octo-02": {
        "gamma_wait_seconds": ("exceeds 300 s", "preempt qg3 and start gamma_preview early", "notify the submitter of the wait"),
        "delta_wait_seconds": ("exceeds 150 s", "preempt qd1 and start delta_preview early", "notify the submitter of the wait"),
        "pool_remaining_memory_mb": ("stays at or above 24576 MB after admitting all three READY pool jobs", "admit all three pool jobs", "defer the smallest pool job"),
        "gpu_waves_required": ("is at most 4", "reserve a single-batch GPU plan", "reserve a multi-wave GPU plan"),
        "effective_timeout_margin_seconds": ("stays at or above 30 s", "let t1 keep running", "restart t1 now"),
        "worst_case_backlog_seconds": ("exceeds 300 s", "escalate r1 to the incident team", "let the remaining retries finish"),
        "initial_transfer_seconds": ("fits within 600 s", "start the pipeline now", "delay the pipeline start"),
        "start_slack_seconds": ("is at least 140 s", "admit d1 this cycle", "defer d1 to the next cycle"),
    },
    "quad-01": {
        "kappa_wait_seconds": ("exceeds 240 s", "preempt qk3 and start kappa_preview early", "notify the submitter of the wait"),
        "pool_remaining_memory_mb": ("stays at or above 12288 MB after admitting all three READY pool jobs", "admit all three pool jobs", "defer the smallest pool job"),
        "effective_timeout_margin_seconds": ("stays at or above 30 s", "let t1 keep running", "restart t1 now"),
        "initial_transfer_seconds": ("fits within 90 s", "start the pipeline now", "delay the pipeline start"),
    },
    "quad-02": {
        "lambda_wait_seconds": ("exceeds 300 s", "preempt ql1 and start lambda_preview early", "notify the submitter of the wait"),
        "pool_remaining_memory_mb": ("stays at or above 16384 MB after admitting all three READY pool jobs", "admit all three pool jobs", "defer the smallest pool job"),
        "effective_timeout_margin_seconds": ("stays at or above 30 s", "let t1 keep running", "restart t1 now"),
        "initial_transfer_seconds": ("fits within 90 s", "start the pipeline now", "delay the pipeline start"),
    },
}


def main():
    for spec in SCENARIOS:
        key = spec["file"].replace("public-multi-", "").replace(".json", "")
        rules = RULES[key]
        tasks, events, pendings, probes = [], [], [], []
        for sub_tasks, sub_events, sub_pending, sub_probe in spec["subs"]:
            tasks.extend(sub_tasks)
            events.extend(sub_events)
            gate = GATES[sub_pending["field"]]
            sub_pending["controller_note"] = (
                f"The plan branch for {gate[0]} depends on this value; it must be "
                "verified before the controller executes the chosen plan."
            )
            pendings.append(sub_pending)
            probes.append(sub_probe)
        rule_texts = []
        for p in pendings:
            cond, plan_t, plan_f = rules[p["field"]]
            rule_texts.append(
                f"if the value of {p['field']} {cond}, the plan is: {plan_t}; "
                f"otherwise: {plan_f}")
        goal = (
            "Choose the next scheduling plan by applying every rule below. Your "
            "action reason must state each chosen plan together with the value "
            "that determines it, and you must delegate one bounded verification "
            "task per value. Rules: " + "; ".join(rule_texts) + ". Preserve "
            "dependencies, resource limits and failure evidence."
        )
        case = {
            "id": spec["id"].replace("multi-", "multi3-"),
            "goal": goal,
            "public_background": {
                "runtime": spec["runtime"],
                "tasks": tasks,
                "pending_decisions": pendings,
                "events": events,
            },
            "current_state": spec["current_state"],
            "probe": {
                "family": "multi_verification_conditional",
                "intended_questions": [p[0] for p in probes],
                "expected_answers": [p[1] for p in probes],
                "derivations": [p[2] for p in probes],
            },
        }
        public_ints = standalone_integers(
            {"goal": case["goal"], "public_background": case["public_background"],
             "current_state": case["current_state"]})
        answers = case["probe"]["expected_answers"]
        for a in answers:
            assert a not in public_ints, f"{case['id']}: answer {a} leaks in public JSON"
        assert len(answers) == len(set(answers)), f"{case['id']}: duplicate answers"
        path = OUT / spec["file"].replace("multi-", "multi3-")
        path.write_text(json.dumps(case, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
        print(f"wrote {path.name}: K={len(answers)}, answers={answers}")
    print(f"total {len(SCENARIOS)} scenarios")


if __name__ == "__main__":
    main()
