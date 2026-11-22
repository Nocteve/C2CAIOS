"""Grade ACB batches (acb1-*): scheduling validity, wait-state accounting,
repair dynamics, and offline Worker-answer audit against probe.expected_answers.

Usage: python scripts/grade_acb_runs.py --batch-dir <dir-with-acb1-*> --out <json>
"""
import argparse
import json
import re
from pathlib import Path


def extract_value(text):
    m = re.findall(r"=\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return float(m[-1])
    m = re.findall(r"-?\d+(?:\.\d+)?", text)
    return float(m[-1]) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for rp in sorted(Path(args.batch_dir).glob("acb1-*/result.json")):
        r = json.loads(rp.read_text(encoding="utf-8"))
        case = r["case_id"]
        rep = r.get("acb_report") or {}
        scen_path = None
        for cand in Path("data/c2c-core-os").glob("public-*.json"):
            s = json.loads(cand.read_text(encoding="utf-8"))
            if s.get("id") == case:
                scen_path = cand
                break
        expected = []
        if scen_path:
            expected = [float(e) for e in
                        json.loads(scen_path.read_text(encoding="utf-8"))["probe"]["expected_answers"]]

        # audit every executed task's final worker output
        tasks_dir = rp.parent / "tasks"
        correct = executed = 0
        audit = []
        if tasks_dir.exists():
            for tp in sorted(tasks_dir.glob("t*.json")):
                arm = json.loads(tp.read_text(encoding="utf-8"))
                executed += 1
                got = None
                try:
                    out = json.loads(arm.get("output", ""))
                    rep_d = out.get("report", {})
                    texts = [str(v) for v in rep_d.values()] if isinstance(rep_d, dict) else [str(rep_d)]
                except Exception:
                    texts = [arm.get("output", "")]
                for t in texts:
                    got = extract_value(t)
                    if got is not None:
                        break
                ok = bool(expected) and got is not None and \
                    any(abs(got - e) < 1e-6 for e in expected)
                correct += int(ok)
                audit.append({"task": tp.stem, "value": got, "audit_ok": ok,
                              "gate": (arm.get("application_gate") or {}).get("accepted")})

        inv = rep.get("invariants") or []
        waits = {"model": 0.0, "fusion": 0.0, "dependency": 0.0, "ready": 0.0}
        for t in (rep.get("tasks") or {}).values():
            waits["model"] += t.get("wait_model_s") or 0.0
            waits["fusion"] += t.get("wait_fusion_s") or 0.0
            waits["dependency"] += t.get("wait_dependency_s") or 0.0
            waits["ready"] += t.get("ready_s") or 0.0
        rows.append({
            "case": case, "status": r["status"],
            "wall_seconds": r.get("wall_seconds"),
            "delegated": rep.get("total"), "done": rep.get("done"),
            "repairs": rep.get("repair_count"),
            "invariants_ok": all(i.get("ok") for i in inv) if inv else None,
            "invariant_count": len(inv),
            "wait_states": waits,
            "worker_correct": correct, "worker_executed": executed,
            "audit": audit,
            "events": len(r.get("scheduler_events") or []),
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{'case':36s} {'stat':10s} {'deleg':>5s} {'done':>4s} {'rep':>3s} "
          f"{'I1I2':>4s} {'Wmodel':>7s} {'Wfusion':>7s} {'Wdep':>6s} {'correct':>7s}")
    for row in rows:
        w = row["wait_states"]
        print(f"{row['case']:36s} {row['status']:10s} {row['delegated']:5d} "
              f"{row['done']:4d} {row['repairs']:3d} "
              f"{str(row['invariants_ok']):>4s} {w['model']:7.2f} {w['fusion']:7.2f} "
              f"{w['dependency']:6.2f} {row['worker_correct']:3d}/{row['worker_executed']:3d}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
