"""Generator for the probe3 'derived intermediate' scenario family.

Design rules (agreed 2026-10-12):
- Each scenario asks for a derived quantity the Planner must compute anyway for
  its own dispatch judgment (admission capacity, timeout margin, post-release
  free resources, critical path, ...).
- The expected answer integer must NOT appear as a standalone integer anywhere
  in the public JSON (goal / public_background / current_state).  A salient
  wrong candidate (a raw component value) is always present as an anchor.
- 4-6 scattered numbers, mixed operations.
- The probe section is offline-only data: run_c2c_core._messages only serialises
  public_background + current_state, so expected_answer never reaches a model.
"""
import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "c2c-core-os"

GOAL = ("Bring the public execution state to a schedulable next step while "
        "preserving task dependencies, resource limits, and failure evidence.")
POLICY = ("A task is runnable only when all dependencies are terminal-success "
          "and its resource request fits the current free capacity.")


def runtime(cpu, gpu, mem, maxp):
    return {"cpu_slots": cpu, "gpu_slots": gpu, "memory_mb": mem,
            "max_parallel_tasks": maxp, "scheduling_policy": POLICY}


def state(free_cpu, free_gpu, free_mem, active, failed=None, extra_note=None):
    note = ("All facts above are public observations. No hidden expected "
            "schedule is supplied. The controller is waiting on the pending "
            "decision value; a bounded verification of that value is the "
            "useful delegation now. Do not claim a blocked task ran.")
    if extra_note:
        note = note + " " + extra_note
    return {"free_cpu_slots": free_cpu, "free_gpu_slots": free_gpu,
            "free_memory_mb": free_mem, "active_tasks": active,
            "failed_tasks": failed or [], "observation_boundary": note}


def probe(intended, answer, derivation):
    return {"family": "derived_intermediate", "intended_question": intended,
            "expected_answer": answer, "derivation": derivation}


SCENARIOS = [
    {
        "file": "public-probe3-timeout-buffer-01.json",
        "id": "os-c2c-probe3-timeout-buffer-01",
        "runtime": runtime(6, 1, 12288, 3),
        "tasks": [
            {"id": "ingest", "state": "SUCCEEDED", "depends_on": [], "cpu": 2, "gpu": 0, "memory_mb": 2048},
            {"id": "transcode", "state": "RUNNING", "depends_on": ["ingest"], "cpu": 2, "gpu": 1, "memory_mb": 3072},
            {"id": "upload", "state": "BLOCKED", "depends_on": ["transcode"], "cpu": 1, "gpu": 0, "memory_mb": 2048},
        ],
        "pending": {
            "field": "effective_timeout_margin_seconds",
            "question": ("According to the most recent heartbeat and the retry "
                         "policy, how many seconds of effective margin remain "
                         "before transcode hits its timeout (remaining time "
                         "minus the reserved retry buffer)?"),
            "controller_note": "The watchdog is blocked until this margin is verified.",
        },
        "events": [
            {"task": "transcode", "event": "heartbeat", "elapsed_s": 48, "timeout_s": 90},
            {"policy": "retry", "event": "reserved_buffer", "buffer_s": 15},
        ],
        "current_state": state(4, 0, 9216, ["transcode"]),
        "probe": probe("Effective timeout margin for transcode (remaining time minus retry buffer).",
                       27, "90 - 48 - 15 = 27."),
    },
    {
        "file": "public-probe3-admission-mem-01.json",
        "id": "os-c2c-probe3-admission-mem-01",
        "runtime": runtime(8, 2, 32768, 4),
        "tasks": [
            {"id": "ingest", "state": "SUCCEEDED", "depends_on": [], "cpu": 2, "gpu": 0, "memory_mb": 2048},
            {"id": "build", "state": "READY", "depends_on": ["ingest"], "cpu": 1, "gpu": 0, "memory_mb": 3072},
            {"id": "test", "state": "READY", "depends_on": ["ingest"], "cpu": 1, "gpu": 0, "memory_mb": 4096},
            {"id": "bundle", "state": "READY", "depends_on": ["ingest"], "cpu": 2, "gpu": 0, "memory_mb": 2048},
            {"id": "ship", "state": "BLOCKED", "depends_on": ["bundle"], "cpu": 1, "gpu": 0, "memory_mb": 2048},
        ],
        "pending": {
            "field": "remaining_memory_after_ready",
            "question": "If every READY task is admitted, what will the remaining free_memory_mb be?",
            "controller_note": "The capacity planner is blocked until this value is verified.",
        },
        "events": [
            {"task": "build", "event": "admission_ready", "reason": "dependency ingest is terminal-success"},
            {"task": "test", "event": "admission_ready", "reason": "dependency ingest is terminal-success"},
            {"task": "bundle", "event": "admission_ready", "reason": "dependency ingest is terminal-success"},
        ],
        "current_state": state(8, 2, 32768, []),
        "probe": probe("Remaining free_memory_mb after admitting every READY task.",
                       23552, "READY: build 3072 + test 4096 + bundle 2048 = 9216; 32768 - 9216 = 23552."),
    },
    {
        "file": "public-probe3-release-admit-01.json",
        "id": "os-c2c-probe3-release-admit-01",
        "runtime": runtime(12, 2, 32768, 4),
        "tasks": [
            {"id": "etl", "state": "RUNNING", "depends_on": [], "cpu": 4, "gpu": 0, "memory_mb": 6144},
            {"id": "train", "state": "READY", "depends_on": ["etl"], "cpu": 2, "gpu": 1, "memory_mb": 3072},
            {"id": "export", "state": "BLOCKED", "depends_on": ["train"], "cpu": 1, "gpu": 0, "memory_mb": 2048},
        ],
        "pending": {
            "field": "remaining_memory_after_release",
            "question": ("When etl finishes and releases its resources, and train "
                         "is then admitted, what will the remaining free_memory_mb be?"),
            "controller_note": "The scheduler is blocked until this projection is verified.",
        },
        "events": [
            {"task": "etl", "event": "progress", "percent": 96},
            {"task": "train", "event": "admission_ready", "reason": "will be terminal-success once etl completes"},
        ],
        "current_state": state(8, 2, 26624, ["etl"]),
        "probe": probe("Remaining free_memory_mb after etl releases and train is admitted.",
                       29696, "26624 + 6144 = 32768; 32768 - 3072 = 29696."),
    },
    {
        "file": "public-probe3-critical-path-01.json",
        "id": "os-c2c-probe3-critical-path-01",
        "runtime": runtime(8, 1, 16384, 4),
        "tasks": [
            {"id": "ingest", "state": "SUCCEEDED", "depends_on": [], "cpu": 1, "gpu": 0, "memory_mb": 1024, "duration_s": 20},
            {"id": "compress", "state": "READY", "depends_on": ["ingest"], "cpu": 2, "gpu": 0, "memory_mb": 2048, "duration_s": 35},
            {"id": "upload", "state": "BLOCKED", "depends_on": ["compress"], "cpu": 1, "gpu": 0, "memory_mb": 1024, "duration_s": 15},
            {"id": "fetch", "state": "SUCCEEDED", "depends_on": [], "cpu": 1, "gpu": 0, "memory_mb": 1024, "duration_s": 40},
            {"id": "parse", "state": "READY", "depends_on": ["fetch"], "cpu": 1, "gpu": 0, "memory_mb": 1024, "duration_s": 10},
        ],
        "pending": {
            "field": "critical_path_seconds",
            "question": ("What is the total duration in seconds of the longest "
                         "dependency chain that ends at a non-success task?"),
            "controller_note": "The resource planner is blocked until the critical path is verified.",
        },
        "events": [
            {"task": "compress", "event": "admission_ready", "reason": "dependency ingest is terminal-success"},
            {"task": "parse", "event": "admission_ready", "reason": "dependency fetch is terminal-success"},
        ],
        "current_state": state(6, 1, 13312, []),
        "probe": probe("Longest dependency chain duration ending at a non-success task.",
                       70, "ingest 20 + compress 35 + upload 15 = 70; fetch 40 + parse 10 = 50; max = 70."),
    },
    {
        "file": "public-probe3-queue-wait-01.json",
        "id": "os-c2c-probe3-queue-wait-01",
        "runtime": runtime(4, 0, 8192, 2),
        "tasks": [
            {"id": "qa", "state": "QUEUED", "depends_on": [], "cpu": 2, "gpu": 0, "memory_mb": 2048, "estimated_duration_s": 120},
            {"id": "lint", "state": "QUEUED", "depends_on": [], "cpu": 1, "gpu": 0, "memory_mb": 1024, "estimated_duration_s": 90},
            {"id": "bench", "state": "QUEUED", "depends_on": [], "cpu": 2, "gpu": 0, "memory_mb": 2048, "estimated_duration_s": 150},
            {"id": "preview", "state": "READY", "depends_on": [], "cpu": 1, "gpu": 0, "memory_mb": 1024},
        ],
        "pending": {
            "field": "queue_wait_seconds",
            "question": ("The runtime admits QUEUED jobs strictly in order ahead of "
                         "preview. How many seconds must preview wait before it can "
                         "start (sum of estimated durations of all jobs ahead of it)?"),
            "controller_note": "The job submitter is blocked until this wait is verified.",
        },
        "events": [
            {"task": "qa", "event": "queued_behind_none", "estimated_duration_s": 120},
            {"queue": "default", "event": "order", "ahead_of_preview": ["qa", "lint", "bench"]},
        ],
        "current_state": state(4, 0, 8192, [], extra_note="preview is behind qa, lint and bench in the admission queue; nothing is running yet."),
        "probe": probe("Seconds preview must wait: sum of estimated durations of queued jobs ahead of it.",
                       360, "qa 120 + lint 90 + bench 150 = 360."),
    },
    {
        "file": "public-probe3-replica-memory-01.json",
        "id": "os-c2c-probe3-replica-memory-01",
        "runtime": runtime(8, 1, 16384, 4),
        "tasks": [
            {"id": "router", "state": "SUCCEEDED", "depends_on": [], "cpu": 1, "gpu": 0, "memory_mb": 1024},
            {"id": "svc", "state": "READY", "depends_on": ["router"], "cpu": 2, "gpu": 0, "memory_mb": 2560, "replicas": 3},
            {"id": "gateway", "state": "BLOCKED", "depends_on": ["svc"], "cpu": 1, "gpu": 0, "memory_mb": 1024},
        ],
        "pending": {
            "field": "remaining_memory_after_replicas",
            "question": "If all replicas of svc are admitted, what will the remaining free_memory_mb be?",
            "controller_note": "The deploy controller is blocked until this value is verified.",
        },
        "events": [
            {"task": "svc", "event": "admission_ready", "replicas": 3, "reason": "dependency router is terminal-success"},
        ],
        "current_state": state(7, 1, 16384, []),
        "probe": probe("Remaining free_memory_mb after admitting all svc replicas.",
                       8704, "svc total 3 x 2560 = 7680; 16384 - 7680 = 8704."),
    },
    {
        "file": "public-probe3-gpu-waves-01.json",
        "id": "os-c2c-probe3-gpu-waves-01",
        "runtime": runtime(8, 2, 16384, 4),
        "tasks": [
            {"id": "render", "state": "READY", "depends_on": [], "cpu": 1, "gpu": 1, "memory_mb": 2048},
            {"id": "sim", "state": "READY", "depends_on": [], "cpu": 1, "gpu": 2, "memory_mb": 2048},
            {"id": "distill", "state": "READY", "depends_on": [], "cpu": 1, "gpu": 1, "memory_mb": 2048},
            {"id": "eval", "state": "READY", "depends_on": [], "cpu": 1, "gpu": 2, "memory_mb": 2048},
        ],
        "pending": {
            "field": "gpu_waves_required",
            "question": ("GPU jobs run one wave at a time, packing each wave into the "
                         "2 gpu_slots. How many waves are required to finish all "
                         "queued GPU jobs?"),
            "controller_note": "The GPU scheduler is blocked until this wave count is verified.",
        },
        "events": [
            {"queue": "gpu", "event": "backlog", "jobs": ["render", "sim", "distill", "eval"]},
        ],
        "current_state": state(4, 2, 8192, []),
        "probe": probe("Waves required to drain the queued GPU jobs with 2 gpu_slots.",
                       3, "GPU demand 1 + 2 + 1 + 2 = 6; 6 / 2 = 3 waves."),
    },
    {
        "file": "public-probe3-retry-wait-01.json",
        "id": "os-c2c-probe3-retry-wait-01",
        "runtime": runtime(4, 0, 8192, 2),
        "tasks": [
            {"id": "sync", "state": "RUNNING", "depends_on": [], "cpu": 1, "gpu": 0, "memory_mb": 1024,
             "attempt": 2, "max_retries": 5, "backoff_s": 40},
            {"id": "publish", "state": "BLOCKED", "depends_on": ["sync"], "cpu": 1, "gpu": 0, "memory_mb": 1024},
        ],
        "pending": {
            "field": "worst_case_backlog_seconds",
            "question": ("If sync keeps failing and consumes all remaining retries, "
                         "how many additional seconds of backoff waiting should the "
                         "controller budget (remaining retries times backoff_s)?"),
            "controller_note": "The incident coordinator is blocked until this budget is verified.",
        },
        "events": [
            {"task": "sync", "event": "retry_scheduled", "attempt": 2, "backoff_s": 40},
        ],
        "current_state": state(3, 0, 7168, ["sync"]),
        "probe": probe("Worst-case additional backoff seconds: remaining retries times backoff_s.",
                       120, "(5 - 2) x 40 = 120."),
    },
    {
        "file": "public-probe3-transfer-01.json",
        "id": "os-c2c-probe3-transfer-01",
        "runtime": runtime(8, 0, 16384, 3),
        "tasks": [
            {"id": "fetchdata", "state": "READY", "depends_on": [], "cpu": 1, "gpu": 0, "memory_mb": 2048},
            {"id": "train", "state": "BLOCKED", "depends_on": ["fetchdata"], "cpu": 4, "gpu": 0, "memory_mb": 6144},
        ],
        "pending": {
            "field": "initial_transfer_seconds",
            "question": ("How many seconds will the initial dataset transfer take in "
                         "total (dataset size divided by link throughput, plus the "
                         "fixed handshake overhead)?"),
            "controller_note": "The pipeline timer is blocked until this duration is verified.",
        },
        "events": [
            {"link": "primary", "event": "profile", "dataset_mb": 4800, "throughput_mb_s": 60, "handshake_overhead_s": 25},
        ],
        "current_state": state(7, 0, 14336, []),
        "probe": probe("Total initial transfer seconds: size / throughput + handshake overhead.",
                       105, "4800 / 60 = 80; 80 + 25 = 105."),
    },
    {
        "file": "public-probe3-reserve-batch-01.json",
        "id": "os-c2c-probe3-reserve-batch-01",
        "runtime": runtime(10, 1, 32768, 4),
        "tasks": [
            {"id": "watchdog", "state": "RUNNING", "depends_on": [], "cpu": 2, "gpu": 0, "memory_mb": 4096},
            {"id": "agg", "state": "READY", "depends_on": [], "cpu": 2, "gpu": 0, "memory_mb": 5120},
            {"id": "join", "state": "READY", "depends_on": [], "cpu": 2, "gpu": 0, "memory_mb": 3072},
            {"id": "archive", "state": "BLOCKED", "depends_on": ["agg"], "cpu": 1, "gpu": 0, "memory_mb": 2048},
        ],
        "pending": {
            "field": "remaining_memory_after_reserve",
            "question": ("After setting aside the failover memory reservation and "
                         "admitting every READY batch job, what will the remaining "
                         "free_memory_mb be?"),
            "controller_note": "The capacity planner is blocked until this value is verified.",
        },
        "events": [
            {"policy": "failover", "event": "reservation", "memory_mb": 4096},
            {"task": "agg", "event": "admission_ready"},
            {"task": "join", "event": "admission_ready"},
        ],
        "current_state": state(8, 1, 28672, ["watchdog"]),
        "probe": probe("Remaining free_memory_mb after the failover reservation and admitting every READY job.",
                       16384, "28672 - 4096 (reserve) - 5120 (agg) - 3072 (join) = 16384."),
    },
    {
        "file": "public-probe3-latest-start-01.json",
        "id": "os-c2c-probe3-latest-start-01",
        "runtime": runtime(8, 1, 16384, 3),
        "tasks": [
            {"id": "report", "state": "READY", "depends_on": [], "cpu": 2, "gpu": 0, "memory_mb": 4096,
             "remaining_work_s": 210, "hard_deadline_s": 540},
        ],
        "pending": {
            "field": "start_slack_seconds",
            "question": ("The wall clock is at 195s and report still needs 210s of "
                         "work before its hard deadline at 540s. How many seconds "
                         "of slack remain before report must start?"),
            "controller_note": "The deadline monitor is blocked until this slack is verified.",
        },
        "events": [
            {"clock": "runtime", "event": "sync", "now_s": 195},
            {"task": "report", "event": "deadline_set", "hard_deadline_s": 540, "remaining_work_s": 210},
        ],
        "current_state": state(6, 1, 12288, []),
        "probe": probe("Seconds of slack before report must start (deadline - now - remaining work).",
                       135, "540 - 195 - 210 = 135."),
    },
    {
        "file": "public-probe3-batch-makespan-01.json",
        "id": "os-c2c-probe3-batch-makespan-01",
        "runtime": runtime(8, 2, 16384, 4),
        "tasks": [
            {"id": "stage1", "state": "READY", "depends_on": [], "cpu": 2, "gpu": 0, "memory_mb": 2048, "duration_s": 45},
            {"id": "stage2", "state": "BLOCKED", "depends_on": ["stage1"], "cpu": 2, "gpu": 0, "memory_mb": 2048, "duration_s": 30},
            {"id": "stage3", "state": "READY", "depends_on": [], "cpu": 2, "gpu": 1, "memory_mb": 2048, "duration_s": 60},
            {"id": "stage4", "state": "BLOCKED", "depends_on": ["stage3"], "cpu": 1, "gpu": 0, "memory_mb": 1024, "duration_s": 20},
        ],
        "pending": {
            "field": "batch_makespan_seconds",
            "question": ("The two pipelines run in parallel and stages inside a "
                         "pipeline run serially. What is the batch makespan in "
                         "seconds, i.e. the later pipeline finish time?"),
            "controller_note": "The batch timer is blocked until this makespan is verified.",
        },
        "events": [
            {"pipeline": "A", "event": "stages", "order": ["stage1", "stage2"]},
            {"pipeline": "B", "event": "stages", "order": ["stage3", "stage4"]},
        ],
        "current_state": state(4, 1, 10240, []),
        "probe": probe("Batch makespan: the later of the two pipeline durations (each the sum of its stages).",
                       80, "A: 45 + 30 = 75; B: 60 + 20 = 80; makespan = 80."),
    },
    {
        "file": "public-probe3-cpu-turnover-01.json",
        "id": "os-c2c-probe3-cpu-turnover-01",
        "runtime": runtime(32, 1, 16384, 6),
        "tasks": [
            {"id": "stuck", "state": "RUNNING", "depends_on": [], "cpu": 14, "gpu": 0, "memory_mb": 2048,
             "event": "health_check_failed"},
            {"id": "rebalance", "state": "READY", "depends_on": [], "cpu": 21, "gpu": 0, "memory_mb": 4096},
        ],
        "pending": {
            "field": "free_cpu_after_turnover",
            "question": ("After evicting the stuck task and admitting the rebalance "
                         "job, what will the remaining free_cpu_slots be?"),
            "controller_note": "The cluster autoscaler is blocked until this value is verified.",
        },
        "events": [
            {"task": "stuck", "event": "health_check_failed", "consecutive_failures": 4},
            {"task": "rebalance", "event": "admission_ready"},
        ],
        "current_state": state(18, 1, 14336, ["stuck"]),
        "probe": probe("Remaining free_cpu_slots after evicting stuck and admitting rebalance.",
                       11, "18 + 14 = 32; 32 - 21 = 11."),
    },
    {
        "file": "public-probe3-stage-handoff-01.json",
        "id": "os-c2c-probe3-stage-handoff-01",
        "runtime": runtime(10, 1, 16384, 4),
        "tasks": [
            {"id": "stageA", "state": "RUNNING", "depends_on": [], "cpu": 2, "gpu": 0, "memory_mb": 3072},
            {"id": "stageB", "state": "READY", "depends_on": ["stageA"], "cpu": 3, "gpu": 1, "memory_mb": 4096},
            {"id": "stageC", "state": "BLOCKED", "depends_on": ["stageB"], "cpu": 1, "gpu": 0, "memory_mb": 1024},
        ],
        "pending": {
            "field": "remaining_memory_after_handoff",
            "question": ("When stageA completes and releases its resources, and "
                         "stageB is then admitted, what will the remaining "
                         "free_memory_mb be?"),
            "controller_note": "The stage orchestrator is blocked until this projection is verified.",
        },
        "events": [
            {"task": "stageA", "event": "progress", "percent": 91},
            {"task": "stageB", "event": "admission_ready", "reason": "will be terminal-success once stageA completes"},
        ],
        "current_state": state(8, 1, 13312, ["stageA"]),
        "probe": probe("Remaining free_memory_mb after stageA releases and stageB is admitted.",
                       12288, "13312 + 3072 = 16384; 16384 - 4096 = 12288."),
    },
    # ---- 2026-10-17 扩题：14 -> 30（同族规则，压噪声）----
    {
        "file": "public-probe3-queue-wait-02.json",
        "id": "os-c2c-probe3-queue-wait-02",
        "runtime": runtime(4, 0, 8192, 2),
        "tasks": [
            {"id": "pack", "state": "QUEUED", "depends_on": [], "cpu": 2, "gpu": 0, "memory_mb": 2048, "estimated_duration_s": 110},
            {"id": "docs", "state": "QUEUED", "depends_on": [], "cpu": 1, "gpu": 0, "memory_mb": 1024, "estimated_duration_s": 95},
            {"id": "fuzz", "state": "QUEUED", "depends_on": [], "cpu": 2, "gpu": 0, "memory_mb": 2048, "estimated_duration_s": 130},
            {"id": "canary", "state": "READY", "depends_on": [], "cpu": 1, "gpu": 0, "memory_mb": 1024},
        ],
        "pending": {
            "field": "queue_wait_seconds",
            "question": ("The runtime admits QUEUED jobs strictly in order ahead of "
                         "canary. How many seconds must canary wait before it can "
                         "start (sum of estimated durations of all jobs ahead of it)?"),
            "controller_note": "The job submitter is blocked until this wait is verified.",
        },
        "events": [
            {"queue": "default", "event": "order", "ahead_of_canary": ["pack", "docs", "fuzz"]},
        ],
        "current_state": state(4, 0, 8192, [], extra_note="canary is behind pack, docs and fuzz in the admission queue; nothing is running yet."),
        "probe": probe("Seconds canary must wait: sum of estimated durations of queued jobs ahead of it.",
                       335, "pack 110 + docs 95 + fuzz 130 = 335."),
    },
    {
        "file": "public-probe3-admission-mem-02.json",
        "id": "os-c2c-probe3-admission-mem-02",
        "runtime": runtime(8, 2, 24576, 4),
        "tasks": [
            {"id": "prep", "state": "SUCCEEDED", "depends_on": [], "cpu": 2, "gpu": 0, "memory_mb": 2048},
            {"id": "scan", "state": "READY", "depends_on": ["prep"], "cpu": 1, "gpu": 0, "memory_mb": 4096},
            {"id": "index", "state": "READY", "depends_on": ["prep"], "cpu": 1, "gpu": 0, "memory_mb": 3072},
            {"id": "seal", "state": "READY", "depends_on": ["prep"], "cpu": 2, "gpu": 0, "memory_mb": 2048},
        ],
        "pending": {
            "field": "remaining_memory_after_ready",
            "question": "If every READY task is admitted, what will the remaining free_memory_mb be?",
            "controller_note": "The capacity planner is blocked until this value is verified.",
        },
        "events": [
            {"task": "scan", "event": "admission_ready", "reason": "dependency prep is terminal-success"},
            {"task": "index", "event": "admission_ready", "reason": "dependency prep is terminal-success"},
            {"task": "seal", "event": "admission_ready", "reason": "dependency prep is terminal-success"},
        ],
        "current_state": state(8, 2, 24576, []),
        "probe": probe("Remaining free_memory_mb after admitting every READY task.",
                       15360, "READY: scan 4096 + index 3072 + seal 2048 = 9216; 24576 - 9216 = 15360."),
    },
    {
        "file": "public-probe3-admission-mem-03.json",
        "id": "os-c2c-probe3-admission-mem-03",
        "runtime": runtime(8, 2, 28672, 4),
        "tasks": [
            {"id": "ingest2", "state": "SUCCEEDED", "depends_on": [], "cpu": 2, "gpu": 0, "memory_mb": 2048},
            {"id": "dedup", "state": "READY", "depends_on": ["ingest2"], "cpu": 2, "gpu": 0, "memory_mb": 5120},
            {"id": "embed", "state": "READY", "depends_on": ["ingest2"], "cpu": 1, "gpu": 0, "memory_mb": 4096},
            {"id": "compact", "state": "READY", "depends_on": ["ingest2"], "cpu": 1, "gpu": 0, "memory_mb": 2048},
        ],
        "pending": {
            "field": "remaining_memory_after_ready",
            "question": "If every READY task is admitted, what will the remaining free_memory_mb be?",
            "controller_note": "The capacity planner is blocked until this value is verified.",
        },
        "events": [
            {"task": "dedup", "event": "admission_ready", "reason": "dependency ingest2 is terminal-success"},
            {"task": "embed", "event": "admission_ready", "reason": "dependency ingest2 is terminal-success"},
            {"task": "compact", "event": "admission_ready", "reason": "dependency ingest2 is terminal-success"},
        ],
        "current_state": state(8, 2, 28672, []),
        "probe": probe("Remaining free_memory_mb after admitting every READY task.",
                       17408, "READY: dedup 5120 + embed 4096 + compact 2048 = 11264; 28672 - 11264 = 17408."),
    },
    {
        "file": "public-probe3-gpu-waves-02.json",
        "id": "os-c2c-probe3-gpu-waves-02",
        "runtime": runtime(8, 2, 16384, 4),
        "tasks": [
            {"id": "trainA", "state": "READY", "depends_on": [], "cpu": 1, "gpu": 2, "memory_mb": 2048},
            {"id": "trainB", "state": "READY", "depends_on": [], "cpu": 1, "gpu": 2, "memory_mb": 2048},
            {"id": "evalC", "state": "READY", "depends_on": [], "cpu": 1, "gpu": 1, "memory_mb": 1024},
            {"id": "evalD", "state": "READY", "depends_on": [], "cpu": 1, "gpu": 1, "memory_mb": 1024},
        ],
        "pending": {
            "field": "gpu_waves_required",
            "question": ("GPU jobs run one wave at a time, packing each wave into the "
                         "2 gpu_slots. How many waves are required to finish all "
                         "queued GPU jobs?"),
            "controller_note": "The GPU scheduler is blocked until this wave count is verified.",
        },
        "events": [
            {"queue": "gpu", "event": "backlog", "jobs": ["trainA", "trainB", "evalC", "evalD"]},
        ],
        "current_state": state(4, 2, 8192, []),
        "probe": probe("Waves required to drain the queued GPU jobs with 2 gpu_slots.",
                       3, "GPU demand 2 + 2 + 1 + 1 = 6; 6 / 2 = 3 waves."),
    },
    {
        "file": "public-probe3-latest-start-02.json",
        "id": "os-c2c-probe3-latest-start-02",
        "runtime": runtime(8, 1, 16384, 3),
        "tasks": [
            {"id": "ledger", "state": "READY", "depends_on": [], "cpu": 2, "gpu": 0, "memory_mb": 4096,
             "remaining_work_s": 190, "hard_deadline_s": 580},
        ],
        "pending": {
            "field": "start_slack_seconds",
            "question": ("The wall clock is at 220s and ledger still needs 190s of "
                         "work before its hard deadline at 580s. How many seconds "
                         "of slack remain before ledger must start?"),
            "controller_note": "The deadline monitor is blocked until this slack is verified.",
        },
        "events": [
            {"clock": "runtime", "event": "sync", "now_s": 220},
            {"task": "ledger", "event": "deadline_set", "hard_deadline_s": 580, "remaining_work_s": 190},
        ],
        "current_state": state(6, 1, 12288, []),
        "probe": probe("Seconds of slack before ledger must start (deadline - now - remaining work).",
                       170, "580 - 220 - 190 = 170."),
    },
    {
        "file": "public-probe3-latest-start-03.json",
        "id": "os-c2c-probe3-latest-start-03",
        "runtime": runtime(8, 1, 16384, 3),
        "tasks": [
            {"id": "invoice", "state": "READY", "depends_on": [], "cpu": 2, "gpu": 0, "memory_mb": 4096,
             "remaining_work_s": 240, "hard_deadline_s": 520},
        ],
        "pending": {
            "field": "start_slack_seconds",
            "question": ("The wall clock is at 150s and invoice still needs 240s of "
                         "work before its hard deadline at 520s. How many seconds "
                         "of slack remain before invoice must start?"),
            "controller_note": "The deadline monitor is blocked until this slack is verified.",
        },
        "events": [
            {"clock": "runtime", "event": "sync", "now_s": 150},
            {"task": "invoice", "event": "deadline_set", "hard_deadline_s": 520, "remaining_work_s": 240},
        ],
        "current_state": state(6, 1, 12288, []),
        "probe": probe("Seconds of slack before invoice must start (deadline - now - remaining work).",
                       130, "520 - 150 - 240 = 130."),
    },
    {
        "file": "public-probe3-timeout-buffer-02.json",
        "id": "os-c2c-probe3-timeout-buffer-02",
        "runtime": runtime(6, 1, 12288, 3),
        "tasks": [
            {"id": "renderx", "state": "RUNNING", "depends_on": [], "cpu": 2, "gpu": 1, "memory_mb": 3072},
            {"id": "mixdown", "state": "BLOCKED", "depends_on": ["renderx"], "cpu": 1, "gpu": 0, "memory_mb": 2048},
        ],
        "pending": {
            "field": "effective_timeout_margin_seconds",
            "question": ("According to the most recent heartbeat and the retry "
                         "policy, how many seconds of effective margin remain "
                         "before renderx hits its timeout (remaining time minus "
                         "the reserved retry buffer)?"),
            "controller_note": "The watchdog is blocked until this margin is verified.",
        },
        "events": [
            {"task": "renderx", "event": "heartbeat", "elapsed_s": 65, "timeout_s": 110},
            {"policy": "retry", "event": "reserved_buffer", "buffer_s": 20},
        ],
        "current_state": state(4, 0, 9216, ["renderx"]),
        "probe": probe("Effective timeout margin for renderx (remaining time minus retry buffer).",
                       25, "110 - 65 - 20 = 25."),
    },
    {
        "file": "public-probe3-retry-wait-02.json",
        "id": "os-c2c-probe3-retry-wait-02",
        "runtime": runtime(4, 0, 8192, 2),
        "tasks": [
            {"id": "mirror", "state": "RUNNING", "depends_on": [], "cpu": 1, "gpu": 0, "memory_mb": 1024,
             "attempt": 3, "max_retries": 6, "backoff_s": 45},
            {"id": "notify", "state": "BLOCKED", "depends_on": ["mirror"], "cpu": 1, "gpu": 0, "memory_mb": 1024},
        ],
        "pending": {
            "field": "worst_case_backlog_seconds",
            "question": ("If mirror keeps failing and consumes all remaining retries, "
                         "how many additional seconds of backoff waiting should the "
                         "controller budget (remaining retries times backoff_s)?"),
            "controller_note": "The incident coordinator is blocked until this budget is verified.",
        },
        "events": [
            {"task": "mirror", "event": "retry_scheduled", "attempt": 3, "backoff_s": 45},
        ],
        "current_state": state(3, 0, 7168, ["mirror"]),
        "probe": probe("Worst-case additional backoff seconds: remaining retries times backoff_s.",
                       135, "(6 - 3) x 45 = 135."),
    },
    {
        "file": "public-probe3-transfer-02.json",
        "id": "os-c2c-probe3-transfer-02",
        "runtime": runtime(8, 0, 16384, 3),
        "tasks": [
            {"id": "preload", "state": "READY", "depends_on": [], "cpu": 1, "gpu": 0, "memory_mb": 2048},
            {"id": "finetune", "state": "BLOCKED", "depends_on": ["preload"], "cpu": 4, "gpu": 0, "memory_mb": 6144},
        ],
        "pending": {
            "field": "initial_transfer_seconds",
            "question": ("How many seconds will the initial dataset transfer take in "
                         "total (dataset size divided by link throughput, plus the "
                         "fixed handshake overhead)?"),
            "controller_note": "The pipeline timer is blocked until this duration is verified.",
        },
        "events": [
            {"link": "backup", "event": "profile", "dataset_mb": 4200, "throughput_mb_s": 70, "handshake_overhead_s": 30},
        ],
        "current_state": state(7, 0, 14336, []),
        "probe": probe("Total initial transfer seconds: size / throughput + handshake overhead.",
                       90, "4200 / 70 = 60; 60 + 30 = 90."),
    },
    {
        "file": "public-probe3-transfer-03.json",
        "id": "os-c2c-probe3-transfer-03",
        "runtime": runtime(8, 0, 16384, 3),
        "tasks": [
            {"id": "collect", "state": "READY", "depends_on": [], "cpu": 1, "gpu": 0, "memory_mb": 2048},
            {"id": "trainbig", "state": "BLOCKED", "depends_on": ["collect"], "cpu": 4, "gpu": 0, "memory_mb": 6144},
        ],
        "pending": {
            "field": "initial_transfer_seconds",
            "question": ("How many seconds will the initial dataset transfer take in "
                         "total (dataset size divided by link throughput, plus the "
                         "fixed handshake overhead)?"),
            "controller_note": "The pipeline timer is blocked until this duration is verified.",
        },
        "events": [
            {"link": "primary", "event": "profile", "dataset_mb": 6300, "throughput_mb_s": 90, "handshake_overhead_s": 25},
        ],
        "current_state": state(7, 0, 14336, []),
        "probe": probe("Total initial transfer seconds: size / throughput + handshake overhead.",
                       95, "6300 / 90 = 70; 70 + 25 = 95."),
    },
    {
        "file": "public-probe3-reserve-batch-02.json",
        "id": "os-c2c-probe3-reserve-batch-02",
        "runtime": runtime(10, 1, 32768, 4),
        "tasks": [
            {"id": "janitor", "state": "RUNNING", "depends_on": [], "cpu": 2, "gpu": 0, "memory_mb": 4096},
            {"id": "rollup", "state": "READY", "depends_on": [], "cpu": 2, "gpu": 0, "memory_mb": 4096},
            {"id": "merge", "state": "READY", "depends_on": [], "cpu": 2, "gpu": 0, "memory_mb": 3072},
            {"id": "vault", "state": "BLOCKED", "depends_on": ["rollup"], "cpu": 1, "gpu": 0, "memory_mb": 2048},
        ],
        "pending": {
            "field": "remaining_memory_after_reserve",
            "question": ("After setting aside the failover memory reservation and "
                         "admitting every READY batch job, what will the remaining "
                         "free_memory_mb be?"),
            "controller_note": "The capacity planner is blocked until this value is verified.",
        },
        "events": [
            {"policy": "failover", "event": "reservation", "memory_mb": 2048},
            {"task": "rollup", "event": "admission_ready"},
            {"task": "merge", "event": "admission_ready"},
        ],
        "current_state": state(8, 1, 22528, ["janitor"]),
        "probe": probe("Remaining free_memory_mb after the failover reservation and admitting every READY job.",
                       13312, "22528 - 2048 (reserve) - 4096 (rollup) - 3072 (merge) = 13312."),
    },
    {
        "file": "public-probe3-cpu-turnover-02.json",
        "id": "os-c2c-probe3-cpu-turnover-02",
        "runtime": runtime(24, 1, 16384, 6),
        "tasks": [
            {"id": "hog", "state": "RUNNING", "depends_on": [], "cpu": 9, "gpu": 0, "memory_mb": 2048},
            {"id": "repack", "state": "READY", "depends_on": [], "cpu": 13, "gpu": 0, "memory_mb": 4096},
        ],
        "pending": {
            "field": "free_cpu_after_turnover",
            "question": ("After evicting the hog task and admitting the repack "
                         "job, what will the remaining free_cpu_slots be?"),
            "controller_note": "The cluster autoscaler is blocked until this value is verified.",
        },
        "events": [
            {"task": "hog", "event": "health_check_failed", "consecutive_failures": 5},
            {"task": "repack", "event": "admission_ready"},
        ],
        "current_state": state(15, 1, 14336, ["hog"]),
        "probe": probe("Remaining free_cpu_slots after evicting hog and admitting repack.",
                       11, "15 + 9 = 24; 24 - 13 = 11."),
    },
    {
        "file": "public-probe3-batch-makespan-02.json",
        "id": "os-c2c-probe3-batch-makespan-02",
        "runtime": runtime(8, 2, 16384, 4),
        "tasks": [
            {"id": "jobA1", "state": "READY", "depends_on": [], "cpu": 2, "gpu": 0, "memory_mb": 2048, "duration_s": 40},
            {"id": "jobA2", "state": "BLOCKED", "depends_on": ["jobA1"], "cpu": 2, "gpu": 0, "memory_mb": 2048, "duration_s": 25},
            {"id": "jobB1", "state": "READY", "depends_on": [], "cpu": 2, "gpu": 1, "memory_mb": 2048, "duration_s": 55},
            {"id": "jobB2", "state": "BLOCKED", "depends_on": ["jobB1"], "cpu": 1, "gpu": 0, "memory_mb": 1024, "duration_s": 15},
        ],
        "pending": {
            "field": "batch_makespan_seconds",
            "question": ("The two pipelines run in parallel and stages inside a "
                         "pipeline run serially. What is the batch makespan in "
                         "seconds, i.e. the later pipeline finish time?"),
            "controller_note": "The batch timer is blocked until this makespan is verified.",
        },
        "events": [
            {"pipeline": "A", "event": "stages", "order": ["jobA1", "jobA2"]},
            {"pipeline": "B", "event": "stages", "order": ["jobB1", "jobB2"]},
        ],
        "current_state": state(4, 1, 10240, []),
        "probe": probe("Batch makespan: the later of the two pipeline durations (each the sum of its stages).",
                       70, "A: 40 + 25 = 65; B: 55 + 15 = 70; makespan = 70."),
    },
    {
        "file": "public-probe3-replica-memory-02.json",
        "id": "os-c2c-probe3-replica-memory-02",
        "runtime": runtime(8, 1, 16384, 4),
        "tasks": [
            {"id": "edge", "state": "SUCCEEDED", "depends_on": [], "cpu": 1, "gpu": 0, "memory_mb": 1024},
            {"id": "api", "state": "READY", "depends_on": ["edge"], "cpu": 2, "gpu": 0, "memory_mb": 3072, "replicas": 2},
            {"id": "mesh", "state": "BLOCKED", "depends_on": ["api"], "cpu": 1, "gpu": 0, "memory_mb": 1024},
        ],
        "pending": {
            "field": "remaining_memory_after_replicas",
            "question": "If all replicas of api are admitted, what will the remaining free_memory_mb be?",
            "controller_note": "The deploy controller is blocked until this value is verified.",
        },
        "events": [
            {"task": "api", "event": "admission_ready", "replicas": 2, "reason": "dependency edge is terminal-success"},
        ],
        "current_state": state(7, 1, 16384, []),
        "probe": probe("Remaining free_memory_mb after admitting all api replicas.",
                       10240, "api total 2 x 3072 = 6144; 16384 - 6144 = 10240."),
    },
    {
        "file": "public-probe3-stage-handoff-02.json",
        "id": "os-c2c-probe3-stage-handoff-02",
        "runtime": runtime(10, 1, 16384, 4),
        "tasks": [
            {"id": "phase1", "state": "RUNNING", "depends_on": [], "cpu": 2, "gpu": 0, "memory_mb": 2048},
            {"id": "phase2", "state": "READY", "depends_on": ["phase1"], "cpu": 3, "gpu": 1, "memory_mb": 4096},
            {"id": "phase3", "state": "BLOCKED", "depends_on": ["phase2"], "cpu": 1, "gpu": 0, "memory_mb": 1024},
        ],
        "pending": {
            "field": "remaining_memory_after_handoff",
            "question": ("When phase1 completes and releases its resources, and "
                         "phase2 is then admitted, what will the remaining "
                         "free_memory_mb be?"),
            "controller_note": "The stage orchestrator is blocked until this projection is verified.",
        },
        "events": [
            {"task": "phase1", "event": "progress", "percent": 94},
            {"task": "phase2", "event": "admission_ready", "reason": "will be terminal-success once phase1 completes"},
        ],
        "current_state": state(8, 1, 12288, ["phase1"]),
        "probe": probe("Remaining free_memory_mb after phase1 releases and phase2 is admitted.",
                       10240, "12288 + 2048 = 14336; 14336 - 4096 = 10240."),
    },
    {
        "file": "public-probe3-release-admit-02.json",
        "id": "os-c2c-probe3-release-admit-02",
        "runtime": runtime(12, 2, 32768, 4),
        "tasks": [
            {"id": "stream", "state": "RUNNING", "depends_on": [], "cpu": 4, "gpu": 0, "memory_mb": 4096},
            {"id": "reduce", "state": "READY", "depends_on": ["stream"], "cpu": 2, "gpu": 1, "memory_mb": 5120},
            {"id": "publish2", "state": "BLOCKED", "depends_on": ["reduce"], "cpu": 1, "gpu": 0, "memory_mb": 2048},
        ],
        "pending": {
            "field": "remaining_memory_after_release",
            "question": ("When stream finishes and releases its resources, and reduce "
                         "is then admitted, what will the remaining free_memory_mb be?"),
            "controller_note": "The scheduler is blocked until this projection is verified.",
        },
        "events": [
            {"task": "stream", "event": "progress", "percent": 97},
            {"task": "reduce", "event": "admission_ready", "reason": "will be terminal-success once stream completes"},
        ],
        "current_state": state(8, 2, 20480, ["stream"]),
        "probe": probe("Remaining free_memory_mb after stream releases and reduce is admitted.",
                       19456, "20480 + 4096 = 24576; 24576 - 5120 = 19456."),
    },
]


def standalone_integers(obj):
    text = json.dumps(obj, ensure_ascii=False)
    return {int(m) for m in re.findall(r"(?<![\d.])(\d+)(?![\d.])", text)}


def main():
    for spec in SCENARIOS:
        case = {
            "id": spec["id"],
            "goal": GOAL,
            "public_background": {
                "runtime": spec["runtime"],
                "tasks": spec["tasks"],
                "pending_decision": spec["pending"],
                "events": spec["events"],
            },
            "current_state": spec["current_state"],
            "probe": spec["probe"],
        }
        public_ints = standalone_integers(
            {"goal": case["goal"], "public_background": case["public_background"],
             "current_state": case["current_state"]})
        answer = case["probe"]["expected_answer"]
        assert answer not in public_ints, f"{spec['id']}: answer {answer} leaks in public JSON"
        assert len(public_ints) >= 4, f"{spec['id']}: only {len(public_ints)} public numbers"
        path = OUT / spec["file"]
        path.write_text(json.dumps(case, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
        print(f"wrote {path.name}: answer={answer}, public_ints={len(public_ints)}")
    print(f"total {len(SCENARIOS)} scenarios")


if __name__ == "__main__":
    main()
