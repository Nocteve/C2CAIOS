"""Grade os-classics ACB isolated runs: per-field verified quality + coverage.

Reads outputs/acbi-osclassics-<fam>-01/result.json + tasks/*.json and grades
each pending_decision field: verified = some task whose Planner command named
the field was accepted by the public-state simulation gate.  Coverage counts
fields the Planner actually delegated.  Nothing here invents values: expected
numbers come from the host simulation in the gate records.

Usage: python scripts/grade_os_classics.py <run_dir> [<run_dir> ...]
"""
import json
import sys
from pathlib import Path


def grade_run(run_dir: Path) -> dict:
    case_file = None
    data_dir = Path(__file__).resolve().parent.parent / "data" / "c2c-core-os"
    for path in data_dir.glob("public-osclassics-*.json"):
        case = json.loads(path.read_text(encoding="utf-8"))
        if case["id"] == json.loads((run_dir / "result.json").read_text(
                encoding="utf-8")).get("case_id", ""):
            case_file = case
            break
    if case_file is None:
        # fall back to filename
        fam = run_dir.name.rsplit("-", 2)[-2]
        case_file = json.loads((data_dir / f"public-osclassics-{fam}-01.json")
                               .read_text(encoding="utf-8"))
    fields = [d["field"] for d in case_file["public_background"]["pending_decisions"]]
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))

    verified = {f: False for f in fields}
    delegated = {f: False for f in fields}
    attempts = {f: 0 for f in fields}
    reasons = {f: [] for f in fields}
    for arm_file in sorted((run_dir / "tasks").glob("t*.json")):
        if arm_file.name.endswith((".out.json", ".meta.json")):
            continue
        arm = json.loads(arm_file.read_text(encoding="utf-8"))
        gate = arm.get("application_gate") or {}
        command = arm.get("command") or ""
        field = gate.get("verified_field")
        if field is None:
            named = [f for f in fields if f in command]
            field = named[0] if named else None
        if field is None:
            continue
        delegated[field] = True
        attempts[field] += 1
        reasons[field].append(str(gate.get("reason"))[:120])
        if gate.get("accepted"):
            verified[field] = True

    coverage = sum(1 for f in fields if delegated[f])
    passed = sum(1 for f in fields if verified[f])
    return {
        "scenario": run_dir.name,
        "fields": fields,
        "delegated": delegated,
        "verified": verified,
        "attempts": attempts,
        "coverage": f"{coverage}/{len(fields)}",
        "verified_count": f"{passed}/{len(fields)}",
        "planner_stages": len(result.get("planner_stages") or []),
        "payload_pool_events": result.get("payload_pool_events", []),
        "payload_mmap": result.get("payload_mmap"),
        "semantic_success": result.get("semantic_success"),
        "wall_seconds": round(result.get("wall_seconds") or 0.0, 1),
    }


def main():
    runs = [Path(a) for a in sys.argv[1:]]
    if not runs:
        base = Path(__file__).resolve().parent.parent / "outputs"
        runs = sorted(base.glob("acbi-osclassics-*"))
    summary = [grade_run(r) for r in runs]
    out = Path("artifacts/osclassics-batch-20261019") / "grading.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    for s in summary:
        print(f"== {s['scenario']}: coverage {s['coverage']}, "
              f"verified {s['verified_count']}, wall {s['wall_seconds']}s")
        for f in s["fields"]:
            mark = "OK " if s["verified"][f] else ("miss" if s["delegated"][f]
                                                   else "no-delegation")
            print(f"   [{mark}] {f} (attempts={s['attempts'][f]})")
    print(f"\nwritten: {out.resolve()}")


if __name__ == "__main__":
    main()
