"""Generator for multi-task amortization scenarios (one Planner analysis, K Workers).

Design rules (2026-10-14):
- One shared runtime state carries K independent subsystems; each subsystem has
  one derived quantity the Planner must verify anyway for its dispatch judgment
  (queue wait, memory after admission, GPU waves, timeout margin, retry backlog,
  transfer time, start slack).  Every quantity is derivable from disjoint public
  evidence, so K Workers consume the SAME Planner analysis with K different short
  commands -- the system configuration where one C2C fusion amortizes against
  per-task practical T2T handoff generation.
- Each expected answer integer must NOT appear as a standalone integer anywhere
  in the public JSON; raw component values are present as anchors.
- The probe section is offline-only data: run_c2c_core._messages only serialises
  public_background + current_state, so expected answers never reach a model.
"""
import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "c2c-core-os"

POLICY = ("A task is runnable only when all dependencies are terminal-success "
          "and its resource request fits the current free capacity.")


def runtime(cpu, gpu, mem, maxp):
    return {"cpu_slots": cpu, "gpu_slots": gpu, "memory_mb": mem,
            "max_parallel_tasks": maxp, "scheduling_policy": POLICY}


def state(free_cpu, free_gpu, free_mem, active, extra_note=None):
    note = ("All facts above are public observations. No hidden expected "
            "values are supplied. The controller is waiting on the pending "
            "decision values; bounded independent verifications of them are "
            "the useful delegation now. Do not claim a blocked task ran.")
    if extra_note:
        note = note + " " + extra_note
    return {"free_cpu_slots": free_cpu, "free_gpu_slots": free_gpu,
            "free_memory_mb": free_mem, "active_tasks": active,
            "failed_tasks": [], "observation_boundary": note}


def queue_sub(prefix, ids_durations, preview):
    """ids_durations: list of (task_id, seconds) queued strictly ahead of preview."""
    tasks = [{"id": tid, "state": "QUEUED", "depends_on": [], "cpu": 2, "gpu": 0,
              "memory_mb": 1024, "estimated_duration_s": dur}
             for tid, dur in ids_durations]
    tasks.append({"id": preview, "state": "READY", "depends_on": [], "cpu": 1,
                  "gpu": 0, "memory_mb": 1024})
    total = sum(d for _, d in ids_durations)
    events = [{"queue": prefix, "event": "order",
               "ahead_of_" + preview: [tid for tid, _ in ids_durations]}]
    field = prefix + "_wait_seconds"
    q = ("The runtime admits QUEUED jobs strictly in order ahead of " + preview +
         ". How many seconds must " + preview + " wait before it can start "
         "(sum of estimated durations of all jobs ahead of it)?")
    deriv = " + ".join(f"{tid} {d}" for tid, d in ids_durations) + f" = {total}."
    pending = {"field": field, "question": q,
               "controller_note": "The submitter is blocked until this wait is verified."}
    probe = ("Seconds " + preview + " must wait: sum of estimated durations of "
             "queued jobs ahead of it.", total, deriv)
    return tasks, events, pending, probe


def memory_sub(ids_sizes, pool_mb):
    """ids_sizes: READY jobs; pool_mb: memory pool size. Answer: pool - sum."""
    tasks = [{"id": tid, "state": "READY", "depends_on": [], "cpu": 2, "gpu": 0,
              "memory_mb": mb} for tid, mb in ids_sizes]
    events = [{"task": tid, "event": "admission_ready",
               "reason": "dependencies are terminal-success"} for tid, _ in ids_sizes]
    total = sum(mb for _, mb in ids_sizes)
    answer = pool_mb - total
    q = ("If every READY job in the pool is admitted, what will the remaining "
         f"free_memory_mb of the {pool_mb} MB pool be?")
    deriv = (f"pool {pool_mb} - (" +
             " + ".join(f"{tid} {mb}" for tid, mb in ids_sizes) +
             f") = {answer}.")
    pending = {"field": "pool_remaining_memory_mb", "question": q,
               "controller_note": "The capacity planner is blocked until this value is verified."}
    probe = ("Remaining pool free_memory_mb after admitting every READY pool job.",
             answer, deriv)
    return tasks, events, pending, probe


def waves_sub(ids_gpus, slots):
    tasks = [{"id": tid, "state": "READY", "depends_on": [], "cpu": 1, "gpu": g,
              "memory_mb": 1024} for tid, g in ids_gpus]
    demand = sum(g for _, g in ids_gpus)
    waves = -(-demand // slots)
    events = [{"queue": "gpu", "event": "backlog",
               "jobs": [tid for tid, _ in ids_gpus], "gpu_slots_per_wave": slots}]
    q = (f"GPU jobs run one wave at a time, packing each wave into the {slots} "
         "gpu_slots. How many waves are required to finish all queued GPU jobs?")
    deriv = f"GPU demand {' + '.join(str(g) for _, g in ids_gpus)} = {demand}; {demand} / {slots} = {waves} waves."
    pending = {"field": "gpu_waves_required", "question": q,
               "controller_note": "The GPU scheduler is blocked until this wave count is verified."}
    probe = ("Waves required to drain the queued GPU jobs.", waves, deriv)
    return tasks, events, pending, probe


def timeout_sub(tid, elapsed, timeout, buffer_s):
    tasks = [{"id": tid, "state": "RUNNING", "depends_on": [], "cpu": 2, "gpu": 0,
              "memory_mb": 2048}]
    answer = timeout - elapsed - buffer_s
    events = [{"task": tid, "event": "heartbeat", "elapsed_s": elapsed, "timeout_s": timeout},
              {"policy": "retry", "event": "reserved_buffer", "buffer_s": buffer_s}]
    q = (f"According to the most recent heartbeat and the retry policy, how many "
         f"seconds of effective margin remain before {tid} hits its timeout "
         "(remaining time minus the reserved retry buffer)?")
    deriv = f"{timeout} - {elapsed} - {buffer_s} = {answer}."
    pending = {"field": "effective_timeout_margin_seconds", "question": q,
               "controller_note": "The watchdog is blocked until this margin is verified."}
    probe = (f"Effective timeout margin for {tid}.", answer, deriv)
    return tasks, events, pending, probe


def retry_sub(tid, attempt, max_retries, backoff):
    tasks = [{"id": tid, "state": "RUNNING", "depends_on": [], "cpu": 1, "gpu": 0,
              "memory_mb": 1024, "attempt": attempt, "max_retries": max_retries,
              "backoff_s": backoff}]
    answer = (max_retries - attempt) * backoff
    events = [{"task": tid, "event": "retry_scheduled", "attempt": attempt, "backoff_s": backoff}]
    q = (f"If {tid} keeps failing and consumes all remaining retries, how many "
         "additional seconds of backoff waiting should the controller budget "
         "(remaining retries times backoff_s)?")
    deriv = f"({max_retries} - {attempt}) x {backoff} = {answer}."
    pending = {"field": "worst_case_backlog_seconds", "question": q,
               "controller_note": "The incident coordinator is blocked until this budget is verified."}
    probe = ("Worst-case additional backoff seconds.", answer, deriv)
    return tasks, events, pending, probe


def transfer_sub(link, dataset_mb, throughput, overhead):
    answer = dataset_mb // throughput + overhead
    events = [{"link": link, "event": "profile", "dataset_mb": dataset_mb,
               "throughput_mb_s": throughput, "handshake_overhead_s": overhead}]
    q = ("How many seconds will the initial dataset transfer take in total "
         "(dataset size divided by link throughput, plus the fixed handshake overhead)?")
    deriv = f"{dataset_mb} / {throughput} = {dataset_mb // throughput}; {dataset_mb // throughput} + {overhead} = {answer}."
    pending = {"field": "initial_transfer_seconds", "question": q,
               "controller_note": "The pipeline timer is blocked until this duration is verified."}
    probe = ("Total initial transfer seconds.", answer, deriv)
    return [], events, pending, probe


def slack_sub(tid, now, work, deadline):
    tasks = [{"id": tid, "state": "READY", "depends_on": [], "cpu": 2, "gpu": 0,
              "memory_mb": 2048, "remaining_work_s": work, "hard_deadline_s": deadline}]
    answer = deadline - now - work
    events = [{"clock": "runtime", "event": "sync", "now_s": now},
              {"task": tid, "event": "deadline_set", "hard_deadline_s": deadline,
               "remaining_work_s": work}]
    q = (f"The wall clock is at {now}s and {tid} still needs {work}s of work "
         f"before its hard deadline at {deadline}s. How many seconds of slack "
         f"remain before {tid} must start?")
    deriv = f"{deadline} - {now} - {work} = {answer}."
    pending = {"field": "start_slack_seconds", "question": q,
               "controller_note": "The deadline monitor is blocked until this slack is verified."}
    probe = (f"Seconds of slack before {tid} must start.", answer, deriv)
    return tasks, events, pending, probe


SCENARIOS = [
    {
        "file": "public-multi-octo-01.json",
        "id": "os-c2c-multi-octo-01",
        "runtime": runtime(32, 4, 65536, 8),
        "subs": [
            queue_sub("alpha", [("qa1", 120), ("qa2", 90), ("qa3", 150)], "alpha_preview"),
            queue_sub("beta", [("qb1", 60), ("qb2", 75)], "beta_preview"),
            memory_sub([("m1", 3072), ("m2", 4096), ("m3", 2048)], 32768),
            waves_sub([("g1", 1), ("g2", 2), ("g3", 1), ("g4", 2)], 2),
            timeout_sub("t1", 48, 90, 15),
            retry_sub("r1", 1, 5, 40),
            transfer_sub("primary", 4800, 60, 25),
            slack_sub("d1", 200, 180, 520),
        ],
        "current_state": state(29, 4, 62464, ["t1", "r1"]),
    },
    {
        "file": "public-multi-octo-02.json",
        "id": "os-c2c-multi-octo-02",
        "runtime": runtime(32, 4, 65536, 8),
        "subs": [
            queue_sub("gamma", [("qg1", 110), ("qg2", 95), ("qg3", 130)], "gamma_preview"),
            queue_sub("delta", [("qd1", 66), ("qd2", 89)], "delta_preview"),
            memory_sub([("m1", 4096), ("m2", 5120), ("m3", 3072)], 40960),
            waves_sub([("g1", 2), ("g2", 2), ("g3", 2), ("g4", 2), ("g5", 2), ("g6", 2)], 2),
            timeout_sub("t1", 62, 150, 18),
            retry_sub("r1", 3, 7, 30),
            transfer_sub("secondary", 7200, 75, 40),
            slack_sub("d1", 240, 200, 610),
        ],
        "current_state": state(29, 4, 62464, ["t1", "r1"]),
    },
    {
        "file": "public-multi-quad-01.json",
        "id": "os-c2c-multi-quad-01",
        "runtime": runtime(16, 2, 32768, 4),
        "subs": [
            queue_sub("kappa", [("qk1", 100), ("qk2", 80), ("qk3", 120)], "kappa_preview"),
            memory_sub([("m1", 3072), ("m2", 4096), ("m3", 2048)], 24576),
            timeout_sub("t1", 55, 120, 20),
            transfer_sub("primary", 3600, 50, 30),
        ],
        "current_state": state(14, 2, 30720, ["t1"]),
    },
    {
        "file": "public-multi-quad-02.json",
        "id": "os-c2c-multi-quad-02",
        "runtime": runtime(16, 2, 32768, 4),
        "subs": [
            queue_sub("lambda", [("ql1", 140), ("ql2", 105), ("ql3", 96)], "lambda_preview"),
            memory_sub([("m1", 2048), ("m2", 4096), ("m3", 2048)], 28672),
            timeout_sub("t1", 47, 100, 12),
            transfer_sub("secondary", 5400, 72, 20),
        ],
        "current_state": state(14, 2, 30720, ["t1"]),
    },
]

GOAL_TEMPLATES = {
    8: ("Before the next scheduling change, produce eight independent verification "
        "reports covering: {items}. The reports are read-only and must preserve "
        "all dependencies and event evidence."),
    4: ("Before the next scheduling change, produce four independent verification "
        "reports covering: {items}. The reports are read-only and must preserve "
        "all dependencies and event evidence."),
}

ITEM_PHRASES = {
    "alpha_wait_seconds": "the alpha queue wait for alpha_preview",
    "beta_wait_seconds": "the beta queue wait for beta_preview",
    "gamma_wait_seconds": "the gamma queue wait for gamma_preview",
    "delta_wait_seconds": "the delta queue wait for delta_preview",
    "kappa_wait_seconds": "the kappa queue wait for kappa_preview",
    "lambda_wait_seconds": "the lambda queue wait for lambda_preview",
    "pool_remaining_memory_mb": "the remaining memory of the pool after admitting every READY pool job",
    "gpu_waves_required": "the number of GPU waves to drain the gpu backlog",
    "effective_timeout_margin_seconds": "the effective timeout margin for t1",
    "worst_case_backlog_seconds": "the worst-case retry backlog for r1",
    "initial_transfer_seconds": "the initial dataset transfer seconds for the profiled link",
    "start_slack_seconds": "the start slack for d1",
}


def standalone_integers(obj):
    text = json.dumps(obj, ensure_ascii=False)
    # A number glued to a letter (task ids like "qa3") is a token, not a
    # standalone integer value.
    return {int(m) for m in re.findall(r"(?<![\d.a-zA-Z])(\d+)(?![\d.])", text)}


def main():
    for spec in SCENARIOS:
        tasks, events, pendings, probes = [], [], [], []
        for sub_tasks, sub_events, sub_pending, sub_probe in spec["subs"]:
            tasks.extend(sub_tasks)
            events.extend(sub_events)
            pendings.append(sub_pending)
            probes.append(sub_probe)
        items = ", ".join(ITEM_PHRASES[p["field"]] for p in pendings)
        case = {
            "id": spec["id"],
            "goal": GOAL_TEMPLATES[len(spec["subs"])].format(items=items),
            "public_background": {
                "runtime": spec["runtime"],
                "tasks": tasks,
                "pending_decisions": pendings,
                "events": events,
            },
            "current_state": spec["current_state"],
            "probe": {
                "family": "multi_verification",
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
            assert a not in public_ints, f"{spec['id']}: answer {a} leaks in public JSON"
        assert len(answers) == len(set(answers)), f"{spec['id']}: duplicate answers"
        assert len(public_ints) >= 4, f"{spec['id']}: only {len(public_ints)} public numbers"
        path = OUT / spec["file"]
        path.write_text(json.dumps(case, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
        print(f"wrote {path.name}: K={len(answers)}, answers={answers}")
    print(f"total {len(SCENARIOS)} scenarios")


if __name__ == "__main__":
    main()
