"""Dispatch-gate redesign of the multi-task amortization scenarios (2026-10-14).

v4 lesson: when the K values are framed as post-hoc "verification reports", the
Planner (per the frozen analysis policy) does not pre-solve them and the 0.6B
Workers parrot formula descriptions -- the quality axis collapses for reasons
unrelated to the communication channel.

probe3 principle: the value must be something the Planner must compute ANYWAY
for its own dispatch judgment.  Here every pending value gates a concrete
control decision (admit vs defer, restart vs let run, escalate vs wait, batch
reservation, deadline admission, submitter notification), and the goal says the
controller cannot issue the next scheduling change until each value is
established.  Same subsystems and answers as gen_probe_multi.py; only the goal
and the controller notes change.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_probe_multi import (  # noqa: E402
    SCENARIOS, runtime, state, GOAL_TEMPLATES, standalone_integers)

OUT = Path(__file__).resolve().parent.parent / "data" / "c2c-core-os"

# Decision-gate wording per subsystem field: the control decision that the value
# gates.  Must not contain any expected answer (asserted below).
GATES = {
    "alpha_wait_seconds": ("submitter notification for alpha_preview",
                           "report the wait to the submitter or offer preemption"),
    "beta_wait_seconds": ("submitter notification for beta_preview",
                          "report the wait to the submitter or offer preemption"),
    "gamma_wait_seconds": ("submitter notification for gamma_preview",
                           "report the wait to the submitter or offer preemption"),
    "delta_wait_seconds": ("submitter notification for delta_preview",
                           "report the wait to the submitter or offer preemption"),
    "kappa_wait_seconds": ("submitter notification for kappa_preview",
                           "report the wait to the submitter or offer preemption"),
    "lambda_wait_seconds": ("submitter notification for lambda_preview",
                            "report the wait to the submitter or offer preemption"),
    "pool_remaining_memory_mb": ("pool admission",
                                 "decide how many READY pool jobs to admit"),
    "gpu_waves_required": ("GPU batch reservation",
                           "reserve the waves in the GPU batch plan"),
    "effective_timeout_margin_seconds": ("watchdog policy for t1",
                                         "decide whether to let t1 keep running or restart it"),
    "worst_case_backlog_seconds": ("incident escalation for r1",
                                   "decide whether to escalate r1 or let the retries finish"),
    "initial_transfer_seconds": ("pipeline start timing",
                                 "decide whether the dataset transfer fits the start gate"),
    "start_slack_seconds": ("deadline admission for d1",
                            "decide whether d1 can still be admitted this cycle"),
}


def main():
    for spec in SCENARIOS:
        tasks, events, pendings, probes = [], [], [], []
        for sub_tasks, sub_events, sub_pending, sub_probe in spec["subs"]:
            tasks.extend(sub_tasks)
            events.extend(sub_events)
            gate = GATES[sub_pending["field"]]
            sub_pending["controller_note"] = (
                f"The {gate[0]} controller cannot {gate[1]} until this value is verified."
            )
            pendings.append(sub_pending)
            probes.append(sub_probe)
        k = len(spec["subs"])
        items = "; ".join(
            f"the {GATES[p['field']][0]} decision needs {GATES[p['field']][1]}"
            for p in pendings)
        goal = (
            f"Issue the next scheduling change: {items}. Each pending value below "
            "gates one of those control decisions, so the controller cannot act "
            "until every value is established. Preserve dependencies, resource "
            "limits and failure evidence."
        )
        case = {
            "id": spec["id"].replace("multi-", "multi2-"),
            "goal": goal,
            "public_background": {
                "runtime": spec["runtime"],
                "tasks": tasks,
                "pending_decisions": pendings,
                "events": events,
            },
            "current_state": spec["current_state"],
            "probe": {
                "family": "multi_verification_dispatch_gate",
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
        path = OUT / spec["file"].replace("multi-", "multi2-")
        path.write_text(json.dumps(case, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
        print(f"wrote {path.name}: K={k}, answers={answers}")
    print(f"total {len(SCENARIOS)} scenarios")


if __name__ == "__main__":
    main()
