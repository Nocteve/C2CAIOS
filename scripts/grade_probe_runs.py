"""Offline grader for c2c-core probe runs.

Reads pulled run directories (each containing result.json) and grades every
Worker arm against the probe family's expected answer, which is offline data
never shown to the models.  Usage:

    python scripts/grade_probe_runs.py <pulled_root> [--scenario-dir data/c2c-core-os]

The grader only inspects: the actual Planner action task text (to flag whether
the delegated question matches the intended probe question) and the raw Worker
arm outputs.  A report counts as correct when the expected integer appears as a
standalone numeric token in the Worker's report text.
"""
import argparse
import json
import re
import sys
from pathlib import Path


def standalone_integers(text):
    return {int(m) for m in re.findall(r"(?<![\d.])(\d+)(?![\d.])", text)}


def grade_run(result_path, scenarios):
    result = json.loads(result_path.read_text(encoding="utf-8"))
    case_id = result.get("case_id")
    scenario = scenarios.get(case_id)
    row = {"case_id": case_id, "status": result.get("status"),
           "planner_action": (result.get("planner") or {}).get("action"),
           "arms": {}}
    if scenario is None:
        row["grader_note"] = "no local probe scenario with this case_id; ungraded"
        return row
    expected = scenario["probe"]["expected_answer"]
    intended = scenario["probe"]["intended_question"]
    command = None
    action = row["planner_action"] or {}
    tasks = action.get("tasks") or []
    if tasks:
        command = tasks[0].get("command", "")
    row["intended_question_matches_command"] = (
        command is not None and command.strip() != ""
        and any(keyword.lower() in command.lower()
                for keyword in re.findall(r"[a-zA-Z_]{5,}", intended)[:6]))
    for name, arm in (result.get("arms") or {}).items():
        if not isinstance(arm, dict) or "output" not in arm:
            continue
        output = arm.get("output", "")
        found = standalone_integers(output)
        row["arms"][name] = {
            "expected": expected,
            "found_integers": sorted(found),
            "correct": expected in found,
            "gate_accepted": (arm.get("application_gate") or {}).get("accepted"),
            "output_preview": output[:300],
        }
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pulled_root", type=Path,
                        help="Local directory of pulled run dirs, each with result.json")
    parser.add_argument("--scenario-dir", type=Path, default=Path("data/c2c-core-os"))
    args = parser.parse_args()

    scenarios = {}
    for path in sorted(args.scenario_dir.glob("public-probe*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        scenarios[data["id"]] = data

    rows = []
    for result_path in sorted(args.pulled_root.glob("*/result.json")):
        rows.append(grade_run(result_path, scenarios))

    summary = {}
    for row in rows:
        for name, arm in row.get("arms", {}).items():
            bucket = summary.setdefault(name, {"correct": 0, "total": 0})
            bucket["total"] += 1
            bucket["correct"] += int(arm["correct"])

    print(json.dumps({"rows": rows, "summary": summary}, ensure_ascii=False, indent=2))
    for name, bucket in summary.items():
        print(f"{name}: {bucket['correct']}/{bucket['total']} correct", file=sys.stderr)


if __name__ == "__main__":
    main()
