"""Grade the A2v2 feedback-loop batch (a2v2loop-*).

Per scenario, per arm (text_full/text_brief/c2c_shared), from result.json:
  - feedback_present: feedback_planner[arm] exists
  - gate_accepted: observation.application_gate.accepted
  - consumption: numbers found in worker report that also appear in the
    feedback-time Planner analysis note (rate = consumed/salient)
  - feedback_action: the Planner's next action after feedback (finish/delegate/inspect)
  - finish_rush: feedback action is finish while the observation carried a
    rejected or contradicted gate
  - repair_after_reject: feedback action is delegate after gate rejection

Writes a JSON summary to --out.
"""
import argparse
import json
import re
from pathlib import Path


def salient_numbers(text):
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for rp in sorted(Path(args.batch_dir).glob("a2v2loop-*/result.json")):
        r = json.loads(rp.read_text(encoding="utf-8"))
        case = r["case_id"]
        fb = r.get("feedback_planner") or {}
        for arm in ("text_full", "text_brief", "c2c_shared"):
            rec = fb.get(arm)
            if not isinstance(rec, dict):
                rows.append({"case": case, "arm": arm, "feedback_present": False})
                continue
            obs = rec.get("observation") or {}
            gate = obs.get("application_gate") or {}
            worker_output = obs.get("worker_output") or ""
            analysis = (rec.get("analysis") or {}).get("output") or ""
            action = rec.get("action")
            if isinstance(action, dict):
                action_name = action.get("action")
            else:
                action_name = None
            sal = salient_numbers(worker_output)
            consumed = {n for n in sal if n in analysis}
            rejected = not gate.get("accepted", False)
            contradictory = bool(gate.get("contradictory_success_claim"))
            rows.append({
                "case": case, "arm": arm, "feedback_present": True,
                "gate_accepted": bool(gate.get("accepted", False)),
                "gate_reason": gate.get("reason"),
                "contradictory": contradictory,
                "salient_numbers": sorted(sal),
                "consumed_numbers": sorted(consumed),
                "consumption_rate": (len(consumed) / len(sal)) if sal else None,
                "feedback_action": action_name,
                "finish_rush": (action_name == "finish" and (rejected or contradictory)),
                "repair_after_reject": (action_name == "delegate" and rejected),
                "feedback_analysis_len": len(analysis),
            })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")

    # aggregate
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(int))
    rates = defaultdict(list)
    for row in rows:
        a = row["arm"]
        agg[a]["n"] += 1
        if row.get("feedback_present"):
            agg[a]["feedback_present"] += 1
        if row.get("gate_accepted"):
            agg[a]["gate_accepted"] += 1
        if row.get("consumption_rate") is not None:
            rates[a].append(row["consumption_rate"])
            if row["consumption_rate"] >= 0.5:
                agg[a]["consumed_half"] += 1
        if row.get("finish_rush"):
            agg[a]["finish_rush"] += 1
        if row.get("repair_after_reject"):
            agg[a]["repair_after_reject"] += 1
        if row.get("feedback_action"):
            agg[a][f"act_{row['feedback_action']}"] += 1
    print(f"{'arm':10s} {'n':>3s} {'fb':>3s} {'gate_ok':>7s} {'消费率≥50%':>9s} {'均消费率':>7s} {'finish抢跑':>8s} {'拒绝后修复':>8s}")
    import statistics
    for a in ("text_full", "text_brief", "c2c_shared"):
        d = agg[a]
        mean_rate = statistics.mean(rates[a]) if rates[a] else float("nan")
        print(f"{a:10s} {d['n']:3d} {d['feedback_present']:3d} {d['gate_accepted']:7d} "
              f"{d['consumed_half']:9d} {mean_rate:7.2f} {d['finish_rush']:8d} {d['repair_after_reject']:8d} "
              f"acts: fin={d.get('act_finish',0)} del={d.get('act_delegate',0)} ins={d.get('act_inspect',0)}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
