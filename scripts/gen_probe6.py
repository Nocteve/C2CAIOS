"""Mixed-domain multi-agent inspection scenarios (probe6, 2026-10-21).

Motivation: demo-ready multi-agent run. One runtime, FOUR pending gauges that
must all be verified before the shift report is published:
  1. OS  - memory accounting: total peak RSS across services (addition chain)
  2. OS  - disk I/O accounting: total sector demand of the pending window
           (addition chain)
  3. non-OS - inventory: net stock of SKU-7 across two warehouses
           (add/sub chain, 4 operands)
  4. non-OS - roster compliance: total crew hours this week (addition chain)

v2 (2026-10-21 late): the v1 gauges were lookup/direction-selection tasks and
the 0.6B Worker failed nearly all of them (probe6 b-g rounds, honest record in
docs/probe6-results-2026-10-21.md).  All four values are now single bounded
ARITHMETIC chains over operands quoted inline in the pending questions -- the
task shape the 0.6B Worker is proven to complete (multi3 / probe4 / probe5
batches).  Every value remains a single bounded step, so the point of the
scene stays Planner-side: four bounded verifications delegated in ONE run.

Leak discipline: expected answers never appear as standalone integers in any
public string; operands live in structured event payloads mixed with
non-operative fields (limits, caps, algorithm names).  All expected values are
re-derived in this file and asserted.
"""
import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "c2c-core-os"

OBSERVATION_BOUNDARY = (
    "All facts above are public observations. No hidden expected values are "
    "supplied. The controller is waiting on the pending decision values; "
    "bounded independent verifications of them are the useful delegation "
    "now. Do not claim a blocked task ran."
)

RUNTIME = {
    "cpu_slots": 16,
    "gpu_slots": 2,
    "memory_mb": 32768,
    "max_parallel_tasks": 4,
    "scheduling_policy": (
        "A task is runnable only when all dependencies are terminal-success "
        "and its resource request fits the current free capacity."
    ),
}


def build(key, mem_peaks, mem_limit, head, direction, queue,
          sku_in_a, sku_out_a, sku_in_b, sku_out_b,
          crew_hours, hour_cap, thresholds):
    # ---- independent re-derivation (asserted) ----
    total_rss = sum(mem_peaks)
    total_sectors = sum(queue)
    net_stock = sku_in_a - sku_out_a + sku_in_b - sku_out_b
    total_hours = sum(crew_hours)
    answers = {"total_peak_rss_mb": total_rss, "total_pending_sectors": total_sectors,
               "sku7_net_stock": net_stock, "total_crew_hours": total_hours}

    svc_ids = [f"svc{i+1}" for i in range(len(mem_peaks))]
    crews = [f"crew_{c}" for c in ["alpha", "bravo", "charlie"][: len(crew_hours)]]

    tasks = [
        {"id": f"vg{i+1}", "state": "QUEUED", "depends_on": [],
         "cpu": 1, "gpu": 0, "memory_mb": 512,
         "estimated_duration_s": 30,
         "purpose": f"verify gauge {f}"}
        for i, f in enumerate(answers)
    ]
    events = [
        {"subsystem": "memory_accounting", "memory_max_mb": mem_limit,
         "service_peak_rss_mb": {s: p for s, p in zip(svc_ids, mem_peaks)}},
        {"subsystem": "disk_scheduling", "algorithm": "SCAN",
         "head_sector": head, "head_direction": direction,
         "pending_request_sectors": queue},
        {"subsystem": "inventory", "sku": "SKU-7",
         "warehouse_a": {"inbound": sku_in_a, "outbound": sku_out_a},
         "warehouse_b": {"inbound": sku_in_b, "outbound": sku_out_b}},
        {"subsystem": "roster_compliance", "legal_week_cap_hours": hour_cap,
         "crew_week_hours": {c: h for c, h in zip(crews, crew_hours)}},
    ]
    pendings = [
        {"field": "total_peak_rss_mb",
         "question": (
             "The memory_accounting record shows service_peak_rss_mb in "
             "service order = " + str(list(mem_peaks)) + ". What is the total "
             "peak RSS across all services (the sum of the listed peaks in "
             "MB)?"),
         "controller_note": (
             "The daily report cannot print the memory footline without this "
             "total; it must be verified before publication.")},
        {"field": "total_pending_sectors",
         "question": (
             "The disk_scheduling record shows pending_request_sectors = " +
             str(list(queue)) + ". What is the total sector demand of the "
             "pending window (the sum of all listed pending sectors)?"),
         "controller_note": (
             "The I/O backlog line of the report is blocked on this total; it "
             "must be verified before publication.")},
        {"field": "sku7_net_stock",
         "question": (
             "The inventory record shows SKU-7 warehouse_a inbound " +
             str(sku_in_a) + " outbound " + str(sku_out_a) +
             ", warehouse_b inbound " + str(sku_in_b) + " outbound " +
             str(sku_out_b) + ". What is the net on-hand stock across both "
             "warehouses (inbound minus outbound, combined)?"),
         "controller_note": (
             "The replenishment line of the report is blocked on this "
             "number; it must be verified before publication.")},
        {"field": "total_crew_hours",
         "question": (
             "The roster_compliance record shows crew_week_hours in crew "
             "order = " + str(list(crew_hours)) + ". What is the total crew "
             "hours logged this week (the sum of the listed hours)?"),
         "controller_note": (
             "The compliance line of the report is blocked on this total; it "
             "must be verified before publication.")},
    ]
    derivations = [
        f"{' + '.join(str(p) for p in mem_peaks)} = {total_rss}.",
        f"{' + '.join(str(s) for s in queue)} = {total_sectors}.",
        f"{sku_in_a} - {sku_out_a} + {sku_in_b} - {sku_out_b} = {net_stock}.",
        f"{' + '.join(str(h) for h in crew_hours)} = {total_hours}.",
    ]
    intended = [
        "Total peak RSS across all services (sum, MB).",
        "Total sector demand of the pending window (sum).",
        "Net on-hand stock of SKU-7 across the two warehouses.",
        "Total crew hours logged this week (sum).",
    ]
    rules = [
        (f"if the value of total_peak_rss_mb is at least {thresholds[0]}, the plan is: page "
         "the on-call SRE and request a restart quota; otherwise: mark the "
         "memory gauge green"),
        (f"if the value of total_pending_sectors stays at or below {thresholds[1]}, the plan "
         "is: schedule the defrag window after the current sweep; otherwise: "
         "hold the defrag"),
        (f"if the value of sku7_net_stock stays below {thresholds[2]} units, the plan is: "
         "trigger a replenishment order; otherwise: skip replenishment this "
         "week"),
        (f"if the value of total_crew_hours is at least {thresholds[3]}, the plan is: "
         "file a compliance review for the shift; otherwise: clear the "
         "roster line"),
    ]
    goal = (
        "Bring the public runtime state to a publishable next step while "
        "preserving the audit trail. The controller is waiting on the four "
        "pending decision values; delegate one bounded verification task per "
        "value. Each delegated command must quote the exact operand values "
        "from its question so the Worker can compute the value from the "
        "command alone."
    )
    case = {
        "id": f"os-c2c-probe6-{key}",
        "goal": goal,
        "public_background": {
            "runtime": RUNTIME,
            "tasks": tasks,
            "pending_decisions": pendings,
            "events": events,
        },
        "current_state": {
            "free_cpu_slots": 14,
            "free_gpu_slots": 2,
            "free_memory_mb": 30720,
            "active_tasks": ["t1"],
            "failed_tasks": [],
            "observation_boundary": OBSERVATION_BOUNDARY,
        },
        "probe": {
            "family": "multi_verification_mixed",
            "intended_questions": intended,
            "expected_answers": list(answers.values()),
            "derivations": derivations,
        },
    }
    # threshold assert (multi3 discipline): no rule threshold equals an
    # answer (the chosen branch must be decidable only after computing it)
    for a, t in zip(answers.values(), thresholds):
        assert a != t, (a, t)

    # leak assert: COMPUTED answers must not appear as a standalone token in
    # public text (all four are sums/differences of quoted operands).
    public = json.dumps(case["public_background"], ensure_ascii=False) + \
        json.dumps(case["current_state"], ensure_ascii=False)
    for a in answers.values():
        assert not re.search(rf"(?<![\d.]){a}(?![\d.])", public), \
            f"computed answer {a} leaks into public text"
    return case


def main():
    cases = [
        ("cluster-01",
         [1024, 8200, 2048, 3960, 600], 4096,
         120, "up", [86, 240, 30, 199],
         240, 85, 160, 122,
         [32, 36, 45], 40, (20000, 400, 200, 150)),
        ("cluster-02",
         [512, 2048, 4608, 1024, 3960], 4096,
         200, "down", [86, 240, 18, 199],
         315, 47, 128, 96,
         [28, 44, 37], 40, (15000, 700, 400, 120)),
    ]
    for spec in cases:
        case = build(*spec)
        out = OUT / f"public-probe6-{spec[0]}.json"
        out.write_text(json.dumps(case, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        print("wrote", out.name, "answers=", case["probe"]["expected_answers"])


if __name__ == "__main__":
    main()
