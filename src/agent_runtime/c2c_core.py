"""One real Planner-to-Worker C2C handoff for the course OS scenario.

This is deliberately a narrow core probe.  It does not run the autonomous
runtime, repair tasks, or claim semantic success.  The Planner still decides
whether and what to delegate.  If it delegates, the same actual publication is
consumed by a native receiver arm and by the fused C2C arm.
"""

import hashlib
import json
import os
import re
from pathlib import Path
from time import perf_counter

from .delegation_contract import SHARED_SYSTEM, object_schema
from .planner_tools import WORKER_ROLE_SYSTEM
from .project import ROOT


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _worker_command(task):
    contract = json.dumps({
        "invocation_role": "Worker",
        "instructions": (
            "Use the shared OS scheduling background only for this local assignment. "
            "The cached Planner analysis and action are background only: never copy "
            "their JSON keys or values as your response and never emit the Planner "
            "action. Return one JSON object shaped as {\"action\":\"report\",\"report\":...} "
            "or {\"action\":\"need_info\",\"question\":...}. Compute the final answer "
            "from the granted public state yourself and include the exact numeric result "
            "in the report; do not return the assignment text or a task template as the "
            "answer. Do not "
            "delegate, claim the parent task is complete, or invent an execution result."
        ),
        "command": task["command"],
        "deliverable": task["deliverable"],
        "success_criteria": task["success_criteria"],
        "granted_files": ["runtime-state.json"],
        "snapshot_version": 0,
        "snapshot_sha256": "public-c2c-core-fixture",
        "dependency_artifacts": [],
        "tool_observations": [],
    }, ensure_ascii=False)
    return (
        "WORKER ASSIGNMENT START. The cached Planner transcript is background only. "
        "Ignore its action, tasks, reason and working note as output instructions. "
        "Perform only this local assignment and return a Worker report or need_info. "
        "WORKER CONTRACT JSON:\n" + contract
    )


OS_CLASSICS_FAMILIES = ("os_classics_sched", "os_classics_banker",
                        "os_classics_paging", "os_classics_fault",
                        "os_classics_schedstep")


def _simulate_os_classics(case):
    """Derive expected decision values from public_background on the spot.

    Pure function: the host re-simulates the OS classic (FCFS/SJF, banker's
    safety, FIFO/LRU paging, agent event consistency) instead of trusting any
    stored answer, so grading is public-state-derived exactly like the
    course-OS encode/rank gate.
    """
    bg = case["public_background"]
    family = case["probe"]["family"]

    def jobs_sjf_waiting(jobs):
        order_fcfs = sorted(jobs, key=lambda j: (j["arrival_seconds"], j["id"]))
        t, fcfs_tt = 0, 0
        for j in order_fcfs:
            t = max(t, j["arrival_seconds"]) + j["burst_seconds"]
            fcfs_tt += t - j["arrival_seconds"]
        done, t, sjf_tt, sjf_w = set(), 0, 0, 0
        while len(done) < len(jobs):
            ready = [j for j in jobs if j["id"] not in done
                     and j["arrival_seconds"] <= t]
            if not ready:
                t = min(j["arrival_seconds"] for j in jobs if j["id"] not in done)
                continue
            nxt = min(ready, key=lambda j: (j["burst_seconds"], j["arrival_seconds"], j["id"]))
            t = max(t, nxt["arrival_seconds"]) + nxt["burst_seconds"]
            sjf_tt += t - nxt["arrival_seconds"]
            sjf_w += t - nxt["arrival_seconds"] - nxt["burst_seconds"]
            done.add(nxt["id"])
        return {"fcfs_total_turnaround_seconds": fcfs_tt,
                "sjf_total_turnaround_seconds": sjf_tt,
                "sjf_total_waiting_seconds": sjf_w}

    def banker_safe(avail, alloc, need, ids):
        work, finished = avail, set()
        while len(finished) < len(ids):
            progressed = False
            for pid in ids:
                if pid in finished:
                    continue
                if need[pid] <= work:
                    work += alloc[pid]
                    finished.add(pid)
                    progressed = True
            if not progressed:
                return 0
        return 1

    def paging_faults(refs, frames):
        fifo_q, fifo_set, fifo_faults = [], set(), 0
        for pg in refs:
            if pg not in fifo_set:
                fifo_faults += 1
                if len(fifo_q) >= frames:
                    fifo_set.discard(fifo_q.pop(0))
                fifo_q.append(pg)
                fifo_set.add(pg)
        lru_age, lru_faults = {}, 0
        for i, pg in enumerate(refs):
            if pg not in lru_age:
                lru_faults += 1
                if len(lru_age) >= frames:
                    del lru_age[min(lru_age, key=lambda k: lru_age[k])]
            lru_age[pg] = i
        return {"fifo_faults": fifo_faults, "lru_faults": lru_faults}

    if family == "os_classics_sched":
        return jobs_sjf_waiting(bg["tasks"])
    if family == "os_classics_schedstep":
        # Stepwise metering: the dispatch order is already decided and
        # public; each value is ONE job's completion along that order
        # (start = max(previous completion, own arrival); a bounded
        # single-task computation instead of a full schedule simulation).
        jobs = {j["id"]: j for j in bg["tasks"]}
        out, prev = {}, 0
        for jid in bg["decided_dispatch_order"]:
            job = jobs[jid]
            start = max(prev, job["arrival_seconds"])
            prev = start + job["burst_seconds"]
            out[f"completion_{jid.lower()}_seconds"] = prev
        return out
    if family == "os_classics_banker":
        procs = bg["tasks"]
        alloc = {p["id"]: p["allocated_units"] for p in procs}
        need = {p["id"]: p["maximum_units"] - p["allocated_units"] for p in procs}
        ids = [p["id"] for p in procs]
        out = {"safe_before_request": banker_safe(bg["resource"]["available_units"],
                                                  alloc, need, ids)}
        req = next((e for e in bg.get("events", [])
                    if e.get("event") == "request_arrived"), None)
        if req:
            pid, units = req["process"], req["requested_units"]
            out["request_grant"] = banker_safe(
                bg["resource"]["available_units"] - units,
                {**alloc, pid: alloc[pid] + units},
                {**need, pid: need[pid] - units}, ids)
        return out
    if family == "os_classics_paging":
        return paging_faults(bg["reference_string"], bg["frames"])
    if family == "os_classics_fault":
        events = bg["agent_events"]
        oom = [i for i, e in enumerate(events)
               if e["exit_code"] == 137 and e["peak_rss_mb"] > bg["memory_max_mb"]]
        return {"oom_agent_index": oom[0] if oom else -1,
                "survived_siblings": sum(1 for e in events if e["exit_code"] == 0)}
    raise ValueError(f"unknown os_classics family: {family}")



def _audit_planner_finish(case, action):
    """Audit a Planner finish summary against the public-state simulation.

    The Planner is never repaired: this is an offline audit that extracts
    each reported value from the finish summary text (last number in the
    field's sentence, e.g. "completion_j1_seconds = 0 + 4 = 4" -> 4) and
    compares it with the value the host re-derives from public facts.
    """
    summary = str((action or {}).get("summary") or "")
    expected = _simulate_os_classics(case)
    fields = [d["field"] for d in case["public_background"]["pending_decisions"]]
    comparisons = {}
    lowered = summary.lower()
    for i, field in enumerate(fields):
        want = expected["request_grant"] if ("request" in field and "grant" in field) else expected.get(field)
        start = lowered.find(field.lower())
        if start < 0:
            comparisons[field] = {"expected": want, "reported": None, "match": False}
            continue
        end = len(summary)
        for other in fields:
            if other == field:
                continue
            pos = lowered.find(other.lower(), start + len(field))
            if pos >= 0:
                end = min(end, pos)
        segment = summary[start:end]
        nums = re.findall(r"-?\d+", segment)
        got = int(nums[-1]) if nums else None
        comparisons[field] = {"expected": want, "reported": got, "match": got == want}
    return {
        "finish_summary_audit": True,
        "field_comparisons": comparisons,
        "all_correct": all(c["match"] for c in comparisons.values()),
        "fields_found": sum(1 for c in comparisons.values() if c["reported"] is not None),
    }

def _worker_json(output):
    """Parse a free-decoded Worker output, tolerating a leading think block.

    Workers decode freely; the host validates the real output afterward.  A
    ``<think>...</think>`` preamble before the JSON contract object is part
    of the actual generation, not an error -- strip it and parse the
    remainder before rejecting.
    """
    text = (output or "").strip()
    for candidate in (text, text.split("</think>")[-1].strip()):
        if not candidate:
            continue
        # Markdown-fenced JSON (```json ... ```): keep the fenced body.
        if "```" in candidate:
            fenced = candidate.split("```")
            for chunk in fenced[1:]:
                body = chunk.split("\n", 1)[-1] if chunk.lstrip().startswith("json") else chunk
                body = body.rsplit("```", 1)[0].strip()
                try:
                    return json.loads(body)
                except Exception:
                    pass
        try:
            return json.loads(candidate)
        except Exception:
            pass
        start = candidate.rfind("{")
        if start >= 0:
            try:
                return json.loads(candidate[start:])
            except Exception:
                pass
    raise ValueError("no JSON contract object in worker output")


def _os_classics_gate(case, generated, command=None):
    """Per-task value gate for the os-classics scenario families.

    Contract: each delegated task verifies ONE pending-decision value.  The
    field under test is the one the Planner named in the task command (the
    Planner's own words, not the Worker's), and the Worker report carries a
    single computed number (``{"action":"report","report":{"value":N}}`` or a
    lone numeric scalar).  The host compares it against the value it
    re-derives from public_background, so echoed numbers or improvised field
    names cannot masquerade as a verification.
    """
    try:
        value = _worker_json(generated.get("output", ""))
    except Exception as exc:
        return {"accepted": False, "reason": f"invalid_worker_json:{type(exc).__name__}",
                "public_state_derived": True}
    text = json.dumps(value, ensure_ascii=False).lower() if isinstance(value, dict) else ""
    positive_success = (
        '"state": "succeeded"' in text
        or ("has completed successfully" in text
            and "not completed successfully" not in text
            and "still in progress" not in text
            and "if it has completed successfully" not in text)
    )
    report = value.get("report") if isinstance(value, dict) else None
    expected = _simulate_os_classics(case)
    fields = [d["field"] for d in case["public_background"]["pending_decisions"]]

    def want_for(field):
        return expected["request_grant"] if ("request" in field and "grant" in field) \
            else expected.get(field)

    # A. Command-anchored per-task mode (ACB paths): the command names the field.
    if command:
        named = [f for f in fields if f in command]
        if not named:
            return {"accepted": False,
                    "reason": "no pending-decision field named in Planner command",
                    "public_state_derived": True,
                    "contradictory_success_claim": positive_success}
        field = named[0]
        got = None
        if isinstance(report, dict):
            if isinstance(report.get("value"), (int, float)) \
                    and not isinstance(report.get("value"), bool):
                got = report["value"]
            else:
                # Field-anchored key match: accept an improvised report key
                # that names the verified field (either direction), so a
                # correct computation in an uncontracted shape still grades.
                for k, v in report.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool) \
                            and (k == field or k in field or field in k):
                        got = v
                        break
                if got is None:
                    numerics = [v for v in report.values()
                                if isinstance(v, (int, float)) and not isinstance(v, bool)]
                    if len(numerics) == 1 and len(report) == 1:
                        got = numerics[0]
        elif isinstance(report, (int, float)) and not isinstance(report, bool):
            got = report
        elif isinstance(report, str):
            try:
                got = json.loads(report)
            except Exception:
                got = None
            if not isinstance(got, (int, float)) or isinstance(got, bool):
                got = None
        want = want_for(field)
        match = got == want
        return {
            "accepted": bool(match) and not positive_success,
            "reason": (f"{field}={got} matches public-state simulation ({want})"
                       if match else
                       f"{field}: worker reported {got!r}, public-state simulation says {want!r}"),
            "public_state_derived": True,
            "verified_field": field,
            "expected": want,
            "reported": got,
            "contradictory_success_claim": positive_success,
        }

    # B. Fallback: report must carry every field by exact name (offline audit).
    contract_ok = isinstance(report, dict) and bool(report)
    comparisons = {}
    for field in fields:
        got = report.get(field) if isinstance(report, dict) else None
        comparisons[field] = {"expected": want_for(field), "reported": got,
                              "match": got == want_for(field)}
    all_match = all(c["match"] for c in comparisons.values())
    accepted = contract_ok and all_match and not positive_success
    return {
        "accepted": accepted,
        "reason": ("all pending-decision values match public-state simulation"
                   if accepted else
                   "worker report contradicts public-state simulation"),
        "public_state_derived": True,
        "contract_ok": contract_ok,
        "contradictory_success_claim": positive_success,
        "field_comparisons": comparisons,
    }


def _os_semantic_gate(case, generated, command=None):
    """Check a Worker report against public OS state before parent acceptance."""
    family = (case.get("probe") or {}).get("family")
    if family in OS_CLASSICS_FAMILIES:
        return _os_classics_gate(case, generated, command)
    try:
        value = json.loads(generated.get("output", ""))
        text = json.dumps(value, ensure_ascii=False).lower()
    except Exception as exc:
        return {"accepted": False, "reason": f"invalid_worker_json:{type(exc).__name__}",
                "public_state_derived": True}
    positive_success = (
        '"state": "succeeded"' in text
        or ("has completed successfully" in text
            and "not completed successfully" not in text
            and "still in progress" not in text
            and "if it has completed successfully" not in text)
    )
    # Contract check: exactly one Worker action with a concrete payload.  A
    # report must carry an actual computed value, not a restated assignment.
    action = value.get("action") if isinstance(value, dict) else None
    if action == "report":
        report = value.get("report")
        contract_ok = report is not None and str(report).strip() != ""
    elif action == "need_info":
        contract_ok = bool(str(value.get("question", "")).strip())
    else:
        contract_ok = False
    tasks = {task["id"]: task for task in case["public_background"]["tasks"]}
    encode_state = tasks.get("encode", {}).get("state")
    rank = tasks.get("rank", {})
    if encode_state is None and not rank:
        # Scenario families without the course-OS encode/rank fixtures: the
        # gate is the contract check plus the false-success-claim check.
        accepted = contract_ok and not positive_success
        return {
            "accepted": accepted,
            "reason": ("worker contract satisfied" if accepted else
                       "worker contract violated or fabricated success claim"),
            "public_state_derived": True,
            "expected_encode_state": None,
            "expected_rank_dependencies": [],
            "contract_ok": contract_ok,
            "contradictory_success_claim": positive_success,
        }
    expected_unfinished = encode_state != "SUCCEEDED"
    positive_success = (
        '"state": "succeeded"' in text
        or ("has completed successfully" in text
            and "not completed successfully" not in text
            and "still in progress" not in text
            and "if it has completed successfully" not in text)
    )
    state_preserved = (
        "encode" in text and expected_unfinished
        and any(marker in text for marker in
                ("running", "not terminal-success", "not completed successfully",
                 "not in a terminal-success"))
        and not positive_success
    )
    dependency_preserved = (
        rank.get("depends_on") == ["encode"] and "rank" in text
        and any(marker in text for marker in
                ("blocked", "unresolved", "not fully resolved", "before proceeding",
                 "not terminal-success"))
    )
    accepted = state_preserved and dependency_preserved
    return {
        "accepted": accepted,
        "reason": "public OS dependency facts preserved" if accepted else
                  "Worker report contradicts public dependency state",
        "public_state_derived": True,
        "expected_encode_state": encode_state,
        "expected_rank_dependencies": rank.get("depends_on", []),
        "state_preserved": state_preserved,
        "dependency_preserved": dependency_preserved,
        "contradictory_success_claim": positive_success,
    }


def _feedback_consumption_evidence(observation, analysis_text, action_text):
    """Return auditable evidence that the second Planner turn used its observation.

    A completed generation alone only proves that the observation was supplied.
    Require the response to refer to the supplied gate result or to the concrete
    Worker report before recording consumption; otherwise the run remains an
    unproven feedback attempt.
    """
    text = (analysis_text or "") + "\n" + (action_text or "")
    lowered = text.lower()
    gate = observation.get("application_gate") or {}
    markers = []
    if gate.get("accepted") is True and any(word in lowered for word in
                                             ("accepted", "preserved", "gate", "worker report")):
        markers.append("gate_accepted_reference")
    if gate.get("accepted") is False and any(word in lowered for word in
                                              ("rejected", "contradict", "gate", "worker report")):
        markers.append("gate_rejected_reference")
    report = str(observation.get("worker_output", "")).lower()
    salient = [word for word in ("encode", "rank", "running", "blocked", "succeeded")
               if word in report and word in lowered]
    if len(salient) >= 2:
        markers.append("worker_fact_overlap:" + ",".join(salient))
    # A generic word such as "gate" can be copied from the instruction.  For
    # this OS fixture, require overlap with at least two concrete public facts
    # before calling the observation consumed.
    return {"consumed": len(salient) >= 2, "markers": markers,
            "salient_fact_overlap": salient}


def _feedback_parent_gate(case, action, observation):
    """Check whether a feedback decision is safe to apply to the parent flow."""
    public_tasks = {item["id"]: item for item in case["public_background"]["tasks"]}
    app_gate = observation.get("application_gate") or {}
    if action.get("action") == "finish":
        terminal = all(item.get("state") == "SUCCEEDED" for item in public_tasks.values())
        accepted = bool(app_gate.get("accepted")) and terminal
        return {
            "accepted": accepted,
            "reason": "all public tasks terminal-success and Worker gate accepted"
            if accepted else "finish rejected: public tasks or Worker gate are incomplete",
            "host_generated_repair": False,
        }
    if action.get("action") == "delegate":
        tasks = action.get("tasks") or []
        valid = bool(tasks)
        invalid_dependencies = []
        for task in tasks:
            for dependency in task.get("depends_on", []):
                if public_tasks.get(dependency, {}).get("state") != "SUCCEEDED":
                    invalid_dependencies.append(dependency)
        if invalid_dependencies:
            valid = False
        return {
            "accepted": valid,
            "reason": "delegation dependencies are terminal-success" if valid else
                      "delegation rejected: dependency is not terminal-success",
            "invalid_dependencies": sorted(set(invalid_dependencies)),
            "host_generated_repair": False,
        }
    return {"accepted": False, "reason": "unknown Planner action", "host_generated_repair": False}


def _core_schema(max_tasks=1):
    text = {"type": "string", "minLength": 1, "maxLength": 700}
    assignment = object_schema({
        "command": text,
        "files": {"type": "array", "items": {"type": "string", "enum": ["runtime-state.json"]},
                  "minItems": 1, "maxItems": 1, "uniqueItems": True},
        "depends_on": {"type": "array", "items": {"type": "string"}, "maxItems": 8, "uniqueItems": True},
        "deliverable": text,
        "success_criteria": text,
    })
    delegate = object_schema({"action": {"type": "string", "enum": ["delegate"]},
                              "tasks": {"type": "array", "items": assignment, "minItems": 1, "maxItems": max_tasks},
                              "reason": text})
    # Optional global workflow declaration: the Planner states the ordering
    # once instead of repeating per-task dependency edges.  Optional, so it
    # is added after object_schema (which makes every property required).
    delegate["properties"]["workflow"] = {"type": "string", "enum": ["sequential"]}
    return {"anyOf": [
        delegate,
        object_schema({"action": {"type": "string", "enum": ["finish"]},
                       "status": {"type": "string", "enum": ["completed", "needs_information"]},
                       "summary": text, "cancel_pending": {"type": "boolean"}}),
    ]}


def _apply_declared_workflow(action, tasks, result=None):
    """Encode a Planner-declared global workflow into per-task depends_on.

    provenance: the Planner itself declares the workflow (a structural
    interface lever distinct from per-task depends_on edges); the host only
    translates the declaration into edges and records it on the result.
    """
    if action.get("workflow") == "sequential":
        for i in range(1, len(tasks)):
            tasks[i]["depends_on"] = [f"t{i}"]
        if isinstance(result, dict):
            result["planner_declared_workflow"] = "sequential"
    return tasks


def _messages(case, *, max_tasks=1, last_decision=None, observation=None, remaining_decisions=2):
    task_limit = "one" if max_tasks == 1 else f"up to {max_tasks}"
    instructions = (
        "You are the 4B Planner for an OS-level Agent Runtime. Decide from the "
        f"public state whether to finish or delegate {task_limit} bounded local "
        "scheduling/diagnosis task. Delegation is optional; do not invent a task "
        "graph or hidden answer. Never delegate a task whose dependencies are "
        "not terminal-success. If delegating, give a short command and a "
        "verifiable deliverable. Return exactly one JSON action."
    )
    # Keep the first Planner prompt byte-for-byte identical to the validated
    # rich-r8 run.  Feedback guidance belongs only to the second decision;
    # changing the initial continuation also changes the practical T2T
    # handoff and can turn a baseline failure into a prompt artifact.
    if isinstance(observation, dict) and "worker_observation" in observation:
        instructions += (
            " If a previous Worker observation is present, compare it with the "
            "public state before deciding; it is evidence, not authority."
        )
    if max_tasks > 1:
        # 2026-10-14: under a multi-task budget the 4B Planner degenerated to
        # field-name slug commands ("calculate_x_seconds"), which left every
        # Worker arm without a computable instruction.  Multi mode therefore
        # demands self-contained commands.  The single-task prompt above stays
        # byte-for-byte identical to the validated probe3 runs.
        instructions += (
            " When delegating, every task command must be a self-contained "
            "imperative sentence naming the exact value to compute and the "
            "public evidence to use; a bare field-name slug is not acceptable."
        )
    content = {
        "invocation_role": "Planner",
        "instructions": instructions,
        "public_background": case["public_background"],
        "current_state": case["current_state"],
        # Interface correction 2026-10-10: remaining_delegations must match the
        # task limit of this decision.  Keeping it at 1 in multi mode anchored
        # the Planner to a single task while the instruction and schema allow
        # two (multi run1 and multi-r1 both stopped at one actual task).
        "budget": {"remaining_decisions": remaining_decisions,
                   "remaining_delegations": max_tasks},
        "planner_working_note": "",
        "last_decision_and_result": ({"action": last_decision, "outcome": observation}
                                      if last_decision is not None else None),
        "context_origin": "Fresh public OS runtime state for a core communication probe.",
    }
    return [{"role": "system", "content": SHARED_SYSTEM},
            {"role": "user", "content": json.dumps(content, ensure_ascii=False)}]


def _run_arm(combined, tokenizers, transcript, receiver_prefix, receiver_cache,
             command, method, selected=None, prepared_cache=None,
             prepared_fusion_seconds=0.0, prepared_fusion_timings=None,
             role_system=WORKER_ROLE_SYSTEM, handoff_reencoded=False):
    import torch
    from .cache_ops import cache_digest, cache_bytes, prefill
    from .kv import PlanningPayload, generate_worker, fuse_context
    from .transcript import worker_ids

    timings = {}
    payload_cache = receiver_cache
    fusion_seconds = 0.0
    if method == "c2c_shared" and prepared_cache is None:
        payload_cache, fusion_seconds = fuse_context(
            combined, transcript.source_cache, receiver_cache,
            token_indices=selected, timings=timings)
    elif method == "c2c_shared":
        payload_cache = prepared_cache
        fusion_seconds = prepared_fusion_seconds
        timings = prepared_fusion_timings or {}
    payload = PlanningPayload(tuple(receiver_prefix), payload_cache,
                              fused=method == "c2c_shared",
                              semantic_token_indices=selected if method == "c2c_shared" else None)
    # 2026-10-17 background-scale experiment: the receiver (Qwen3-0.6B, 32k
    # context) must actually ingest an inflated shared background in the text
    # arms; the budget is a hard no-silent-truncation ceiling, not a scenario
    # size limit. Hidden arms are unaffected (short command template only).
    ids = worker_ids(tokenizers["receiver"], list(receiver_prefix), command,
                     32768, role_system=role_system)
    started = perf_counter()
    worker_method = {"text_full": "text_full_cached", "text_brief": "text_brief_cached",
                     "c2c_shared": "c2c_shared"}[method]
    generated = generate_worker(
        combined, tokenizers, ids, payload,
        worker_method, 320)
    generated.update({
        "input_ids": ids,
        "command": command,
        "worker_role_boundary": (
            "native_worker_system_after_published_prefix"
            if role_system is not None else "native_worker_system_inside_published_prefix"),
        "worker_consumed_fused_cache": method == "c2c_shared",
        "source_cache_was_reencoded": False,
        "handoff_text_was_reencoded": handoff_reencoded,
        "source_cache_sha256": cache_digest(transcript.source_cache),
        "receiver_cache_sha256": cache_digest(payload_cache),
        "receiver_cache_bytes": cache_bytes(payload_cache),
        "fusion_seconds": fusion_seconds,
        "fusion_timings": timings,
        "fusion_reused_published_cache": method == "c2c_shared" and prepared_cache is not None,
        "arm_wall_seconds": perf_counter() - started,
        "semantic_status": "unjudged_pending_offline_audit",
    })
    return generated, payload_cache


def _run_hidden_arm(combined, tokenizers, transcript, receiver, command, selection, name):
    """Worker reads only the short command text; Planner KV arrives as hidden fused context.

    The published receiver cache covers [hidden fused Planner positions] +
    [native command-template positions].  No Planner transcript token appears
    in the Worker input ids: the text channel for the Planner output is fully
    closed.  The fuser reference input at hidden positions is the receiver's
    OWN encoding of the actual selected source text (the transcript tokens at
    the selected positions), so both fuser inputs are genuine encodings of the
    same text, matching the fuser's trained fusion pattern.  Only the fused KV
    survives into the payload; the hidden text tokens never enter the Worker's
    visible input.
    """
    from .cache_ops import cache_digest, cache_bytes, extend_past, prefill, slice_cache
    from .kv import PlanningPayload, fuse_context, generate_worker
    from .transcript import worker_ids

    selected = tuple(sorted(selection))
    if not selected:
        raise ValueError("Hidden arm requires a nonempty actual selection")
    sliced_source = slice_cache(transcript.source_cache, selected)
    hidden_text_ids = [transcript.prefix_ids[i] for i in selected]
    hidden_tokens = len(selected)
    receiver_tok = tokenizers["receiver"]
    reference_cache, reference_seconds = prefill(receiver, hidden_text_ids)
    timings = {}
    hidden_cache, fusion_seconds = fuse_context(
        combined, sliced_source, reference_cache,
        token_indices=tuple(range(hidden_tokens)), timings=timings)
    cmd_ids = worker_ids(receiver_tok, [], command, 8192, role_system=WORKER_ROLE_SYSTEM)
    extend_started = perf_counter()
    payload_cache = extend_past(receiver, cmd_ids[:-1], hidden_cache)
    extension_seconds = perf_counter() - extend_started
    payload = PlanningPayload(tuple(cmd_ids[:-1]), payload_cache, fused=True,
                              hidden_context_tokens=hidden_tokens)
    started = perf_counter()
    generated = generate_worker(combined, tokenizers, cmd_ids, payload, "c2c_hidden", 320)
    generated.update({
        "input_ids": cmd_ids,
        "command": command,
        "arm": name,
        "worker_visible_text": "short command template only; Planner transcript text fully excluded",
        "hidden_context_tokens": hidden_tokens,
        "hidden_selection_source_indices": list(selected),
        "hidden_reference_text_ids": hidden_text_ids,
        "hidden_reference_is_receiver_encoding_of_same_text": True,
        "worker_role_boundary": "native_worker_system_command_template_only",
        "worker_consumed_fused_cache": True,
        "source_cache_was_reencoded": False,
        "handoff_text_was_reencoded": False,
        "source_cache_sha256": cache_digest(transcript.source_cache),
        "receiver_cache_sha256": cache_digest(payload_cache),
        "receiver_cache_bytes": cache_bytes(payload_cache),
        "reference_prefill_seconds": reference_seconds,
        "hidden_extension_seconds": extension_seconds,
        "fusion_seconds": fusion_seconds,
        "fusion_timings": timings,
        "arm_wall_seconds": perf_counter() - started,
        "semantic_status": "unjudged_pending_offline_audit",
    })
    return generated


def _ablation_payload(combined, tokenizers, transcript, receiver, command, selection,
                    case=None):
    """Build a hidden-arm fused payload for one selection WITHOUT generating.

    Same provenance as _run_hidden_arm (slice source cache at selected
    positions, receiver prefill of the same text ids, fuser fusion, command
    extension); returned so the caller can both greedily decode and
    teacher-force score under the identical published cache.

    case: when supplied, the receiver prefix is the state-native context
    (public_background + current_state + short command read natively by
    the Worker, per the PPT receiver design) with the hidden fused
    analysis KV prepended -- the same construction as _run_state_arm.
    When None the legacy command-only prefix is kept (validated A3
    ablation path).
    """
    from .cache_ops import cache_digest, cache_bytes, extend_past, prefill, slice_cache
    from .kv import PlanningPayload, fuse_context
    from .transcript import worker_ids

    selected = tuple(sorted(selection))
    if not selected:
        raise ValueError("Ablation requires a nonempty actual selection")
    sliced_source = slice_cache(transcript.source_cache, selected)
    hidden_text_ids = [transcript.prefix_ids[i] for i in selected]
    reference_cache, reference_seconds = prefill(receiver, hidden_text_ids)
    hidden_cache, fusion_seconds = fuse_context(
        combined, sliced_source, reference_cache,
        token_indices=tuple(range(len(selected))))
    if case is not None:
        cmd_ids = _worker_state_ids(tokenizers["receiver"], case, command)
    else:
        cmd_ids = worker_ids(tokenizers["receiver"], [], command, 8192,
                             role_system=WORKER_ROLE_SYSTEM)
    payload_cache = extend_past(receiver, cmd_ids[:-1], hidden_cache)
    payload = PlanningPayload(tuple(cmd_ids[:-1]), payload_cache, fused=True,
                              hidden_context_tokens=len(selected))
    return {
        "payload": payload, "cmd_ids": cmd_ids, "selection": list(selected),
        "hidden_text_ids": hidden_text_ids,
        "reference_prefill_seconds": reference_seconds,
        "fusion_seconds": fusion_seconds,
        "source_cache_sha256": cache_digest(transcript.source_cache),
        "receiver_cache_sha256": cache_digest(payload_cache),
        "receiver_cache_bytes": cache_bytes(payload_cache),
    }



def _span_token_indices(text, lo, hi, tokenizer):
    """Map a character span to token indices fully inside it (same policy as scope)."""
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return [i for i, (a, b) in enumerate(encoded["offset_mapping"]) if a >= lo and b <= hi and a < b]


def _worker_state_ids(receiver_tok, case, command):
    """Worker-native input: public state text + short command, no Planner transcript.

    This is the faithful PPT/T-C2C receiver context: the problem statement
    (public OS state) is read natively by the Worker, exactly as the solver in
    the paper's T-C2C reads the problem; only the Planner analysis is (optionally)
    carried by the C2C channel instead of as text.
    """
    content = {
        "invocation_role": "Worker",
        "context": "Fresh public OS runtime state for the delegated Worker task.",
        "public_background": case["public_background"],
        "current_state": case["current_state"],
    }
    messages = [
        {"role": "system", "content": WORKER_ROLE_SYSTEM},
        {"role": "user", "content": json.dumps(content, ensure_ascii=False)},
        {"role": "user", "content": "WORKER ASSIGNMENT: " + command},
    ]
    return receiver_tok.apply_chat_template(messages, tokenize=True,
                                            add_generation_prompt=True)


def _run_state_arm(combined, tokenizers, case, transcript, receiver, command,
                   selection, name):
    """Worker reads public state + short command natively; optional hidden analysis KV.

    selection=None  -> state_native baseline (no Planner content at all).
    selection=span  -> the same native context with the fused Planner analysis
                       KV prepended as hidden context (c2c_state_hidden); no
                       analysis text ever appears in the Worker input ids.
    """
    import torch
    from .cache_ops import cache_digest, cache_bytes, extend_past, prefill, slice_cache
    from .kv import PlanningPayload, fuse_context, generate_worker

    receiver_tok = tokenizers["receiver"]
    state_ids = _worker_state_ids(receiver_tok, case, command)
    native_cache, native_seconds = prefill(receiver, state_ids[:-1])
    timings = {}
    hidden_tokens = 0
    fusion_seconds = 0.0
    reference_seconds = 0.0
    extension_seconds = 0.0
    selected_positions = []
    if selection is None:
        payload_cache = native_cache
        payload = PlanningPayload(tuple(state_ids[:-1]), payload_cache, fused=False)
    else:
        selected = tuple(sorted(selection))
        sliced_source = slice_cache(transcript.source_cache, selected)
        hidden_text_ids = [transcript.prefix_ids[i] for i in selected]
        hidden_tokens = len(selected)
        selected_positions = list(selected)
        reference_cache, reference_seconds = prefill(receiver, hidden_text_ids)
        hidden_cache, fusion_seconds = fuse_context(
            combined, sliced_source, reference_cache,
            token_indices=tuple(range(hidden_tokens)), timings=timings)
        extend_started = perf_counter()
        payload_cache = extend_past(receiver, state_ids[:-1], hidden_cache)
        extension_seconds = perf_counter() - extend_started
        torch.cuda.synchronize()
        payload = PlanningPayload(tuple(state_ids[:-1]), payload_cache, fused=True,
                                  hidden_context_tokens=hidden_tokens)
    started = perf_counter()
    worker_method = "text_full_cached" if selection is None else "c2c_hidden"
    generated = generate_worker(combined, tokenizers, state_ids, payload, worker_method, 320)
    generated.update({
        "input_ids": state_ids,
        "command": command,
        "arm": name,
        "worker_visible_text": (
            "public state + short command only; no Planner transcript text"
            if selection is None else
            "public state + short command; Planner analysis arrives only as hidden fused KV"),
        "hidden_context_tokens": hidden_tokens,
        "hidden_selection_source_indices": selected_positions,
        "worker_role_boundary": "native_worker_system_public_state_and_command",
        "worker_consumed_fused_cache": selection is not None,
        "source_cache_was_reencoded": False,
        "handoff_text_was_reencoded": False,
        "native_state_prefill_seconds": native_seconds,
        "source_cache_sha256": cache_digest(transcript.source_cache),
        "receiver_cache_sha256": cache_digest(payload_cache),
        "receiver_cache_bytes": cache_bytes(payload_cache),
        "reference_prefill_seconds": reference_seconds,
        "hidden_extension_seconds": extension_seconds,
        "fusion_seconds": fusion_seconds,
        "fusion_timings": timings,
        "arm_wall_seconds": perf_counter() - started,
        "semantic_status": "unjudged_pending_offline_audit",
    })
    return generated


def _run_command_arm(combined, tokenizers, transcript, receiver, receiver_cache,
                     command, task):
    """Worker reads the short command text; the command positions carry C2C-fused KV.

    This is position-matched fusion on the shared dispatch text: the Planner's
    KV at the positions where it generated the dispatch strings (command,
    deliverable, success_criteria) is projected through the fuser and
    scattered onto the receiver cache positions where the IDENTICAL strings
    occur inside the Worker's command contract template.  The Worker's visible
    text is unchanged; the representation of the text it reads is upgraded to
    the Planner's understanding of the same dispatch.  Both fuser inputs are
    genuine encodings of the same transcript (Planner source cache and the
    receiver's native prefill of the published prefix).
    """
    import torch
    from .cache_ops import cache_digest, cache_bytes, clone_cache, prefill
    from .communication_probe import value_spans
    from .kv import PlanningPayload, fuse_context, generate_worker
    from .transcript import worker_ids

    sender_tok, receiver_tok = tokenizers["sender"], tokenizers["receiver"]
    cmd_ids = worker_ids(receiver_tok, [], command, 8192, role_system=WORKER_ROLE_SYSTEM)
    template_text = receiver_tok.decode(cmd_ids, skip_special_tokens=False)
    template_ids = receiver_tok(template_text, add_special_tokens=False)["input_ids"]
    if template_ids != list(cmd_ids):
        raise ValueError("Command template decode/re-encode identity differs")

    generated_text = sender_tok.decode(transcript.generated_ids, skip_special_tokens=False)
    prefixes = [sender_tok.decode(transcript.generated_ids[:i], skip_special_tokens=False)
                for i in range(len(transcript.generated_ids) + 1)]
    if any(not generated_text.startswith(p) for p in prefixes):
        raise ValueError("Actual generated-token text has ambiguous partial decoding")
    lengths = list(map(len, prefixes))
    if lengths != sorted(lengths):
        raise ValueError("Nonmonotonic actual token character boundaries")
    prompt_len = len(transcript.prompt_ids)

    spans = value_spans(transcript.output)
    fields = []
    for key in ("command", "deliverable", "success_criteria"):
        fields.append((key, ("tasks", 0, key)))
    mapping = {}
    target_positions = set()
    for label, path in fields:
        if path not in spans:
            continue
        lo, hi = spans[path]
        if hi - lo < 2 or transcript.output[lo] != '"' or transcript.output[hi - 1] != '"':
            continue
        # Match the raw JSON string content (escaped form): it is spelled
        # identically inside the Planner's generated action JSON and inside
        # the Worker's command contract JSON built from the same task.
        value = transcript.output[lo + 1:hi - 1]
        if not value.strip():
            continue
        if generated_text.count(value) != 1 or template_text.count(value) != 1:
            raise ValueError(f"Dispatch value for {label} is not unique in both texts")
        glo = generated_text.index(value)
        tlo = template_text.index(value)
        src_tokens = [prompt_len + i for i, (a, b) in enumerate(zip(lengths, lengths[1:]))
                      if glo <= a < b <= glo + len(value)]
        tgt_tokens = _span_token_indices(template_text, tlo, tlo + len(value), receiver_tok)
        if not src_tokens or len(src_tokens) != len(tgt_tokens):
            raise ValueError(
                f"Dispatch span {label}: source {len(src_tokens)} tokens vs target "
                f"{len(tgt_tokens)} tokens; position fusion requires exact alignment")
        mapping[label] = {
            "value": value[:200],
            "source_token_span": src_tokens,
            "target_token_span": [i for i in tgt_tokens],
        }
        target_positions.update(tgt_tokens)
    if not mapping:
        raise ValueError("No dispatch value spans available for command fusion")

    timings = {}
    projected_full, fusion_seconds = fuse_context(
        combined, transcript.source_cache, receiver_cache, timings=timings)
    native_cmd, native_seconds = prefill(receiver, cmd_ids[:-1])
    fused_cmd = clone_cache(native_cmd)
    pairs = [(s, t) for label in mapping
             for s, t in zip(mapping[label]["source_token_span"],
                             mapping[label]["target_token_span"])]
    for layer in range(len(fused_cmd)):
        for kv_index in range(2):
            target_tensor = fused_cmd[layer][kv_index]
            source_tensor = projected_full[layer][kv_index]
            for src, tgt in pairs:
                target_tensor[:, :, tgt, :] = source_tensor[:, :, src, :]
    torch.cuda.synchronize()
    payload = PlanningPayload(tuple(cmd_ids[:-1]), fused_cmd, fused=True)
    started = perf_counter()
    generated = generate_worker(combined, tokenizers, cmd_ids, payload, "c2c_command", 320)
    generated.update({
        "input_ids": cmd_ids,
        "command": command,
        "arm": "c2c_command",
        "worker_visible_text": "short command template; the text it reads carries fused Planner dispatch KV at the identical string positions",
        "command_position_mapping": mapping,
        "command_positions_fused": sorted(target_positions),
        "command_positions_fused_count": len(target_positions),
        "command_fusion_is_position_matched_scatter": True,
        "worker_role_boundary": "native_worker_system_command_template_only",
        "worker_consumed_fused_cache": True,
        "source_cache_was_reencoded": False,
        "handoff_text_was_reencoded": False,
        "source_cache_sha256": cache_digest(transcript.source_cache),
        "receiver_cache_sha256": cache_digest(fused_cmd),
        "receiver_cache_bytes": cache_bytes(fused_cmd),
        "native_command_prefill_seconds": native_seconds,
        "fusion_seconds": fusion_seconds,
        "fusion_timings": timings,
        "arm_wall_seconds": perf_counter() - started,
        "semantic_status": "unjudged_pending_offline_audit",
    })
    return generated


def run_c2c_core(output_dir, *, public_filename="public.json", max_tasks=1,
                 rich_planning=False, feedback_loop=False, hidden_arms=False,
                 arith_scaffold=False, hidden_ablate=False, planner_action_tokens=None):
    """Run the narrow core probe on an authorized GPU host."""
    import torch
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import (
        build_token_enforcer_tokenizer_data,
        build_transformers_prefix_allowed_tokens_fn,
    )
    from .cache_ops import cache_digest, cache_bytes, prefill
    from .loading import load_models
    from .communication_probe import state_planning_scope
    from .planning_policy import (ACTION_MESSAGE, analysis_messages,
                                  ARITHMETIC_SCAFFOLD_INSTRUCTION,
                                  FEEDBACK_REVIEW_INSTRUCTION)
    from .transcript import capture_plan

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    public_path = ROOT / "data/c2c-core-os" / public_filename
    case = json.loads(public_path.read_text(encoding="utf-8"))
    config = json.loads((ROOT / "configs/cloud.json").read_text())
    lock = json.loads((ROOT / "configs/cloud.models.lock.json").read_text())
    result = {
        "status": "running", "scope": "one actual Planner decision with full-text, T2T and C2C Worker arms"
        + ("; extra fused arms: hidden-context injection (Worker reads only the short command text) and command-position fusion (the command text the Worker reads carries fused Planner dispatch KV at the identical string positions)"
           if hidden_arms else "")
        + ("; state arms: Worker reads the public state text natively (state_native baseline and c2c_state_hidden with the fused Planner analysis KV as hidden context)"
           if hidden_arms else ""),
        "planner_rich_planning": rich_planning,
        "planner_feedback_loop": feedback_loop,
        "planner_arith_scaffold": arith_scaffold,
        "public_sha256": _sha(public_path), "config": config, "model_pins": lock,
        "source_sha256": {
            public_filename: _sha(public_path),
            "c2c_core.py": _sha(ROOT / "src/agent_runtime/c2c_core.py"),
            "scripts/run.py": _sha(ROOT / "scripts/run.py"),
            "src/agent_runtime/kv.py": _sha(ROOT / "src/agent_runtime/kv.py"),
            "src/agent_runtime/transcript.py": _sha(ROOT / "src/agent_runtime/transcript.py"),
            "src/agent_runtime/planning_policy.py": _sha(ROOT / "src/agent_runtime/planning_policy.py"),
            "src/agent_runtime/planner_tools.py": _sha(ROOT / "src/agent_runtime/planner_tools.py"),
            "src/agent_runtime/communication_probe.py": _sha(ROOT / "src/agent_runtime/communication_probe.py"),
            "src/agent_runtime/cache_ops.py": _sha(ROOT / "src/agent_runtime/cache_ops.py"),
        },
        "pid": os.getpid(), "model_service_separate_process": False,
        "case_id": case["id"], "planner": None, "arms": {},
        "semantic_success": False,
        "semantic_status": "requires independent offline audit of raw Worker outputs",
    }
    started = perf_counter()
    try:
        combined, tokenizers = load_models(config, lock)
        combined.multi_source_fusion_mode = "sequential"
        sender, receiver = combined.model_list[1], combined.model_list[0]
        schema = _core_schema(max_tasks=max_tasks)
        # 8 task objects need far more than 256 action tokens; scale the budget
        # with the task ceiling so the JSON schema constraint, not the token cap,
        # is the binding limit.  Single/two-task prompts stay at 256 byte-for-byte.
        action_max_tokens = 256 if max_tasks <= 2 else 128 * max_tasks
        if planner_action_tokens is not None:
            action_max_tokens = int(planner_action_tokens)
            if action_max_tokens < 256:
                raise ValueError("Explicit action budget must be at least 256")
        stages = []
        transcript = action = planner_messages = None
        for turn in range(2):
            planner_messages = _messages(
                case, max_tasks=max_tasks,
                last_decision=(stages[-1]["action"] or {}).get("action") if stages else None,
                observation=(
                    {"public_observation": "The public OS state and event log were inspected; no hidden files or reference answer were read."}
                    if stages and (stages[-1]["action"] or {}).get("action") == "inspect" else None
                ),
                remaining_decisions=2 - turn,
            )
            planner_tokenizer_data = build_token_enforcer_tokenizer_data(tokenizers["sender"])
            allowed = build_transformers_prefix_allowed_tokens_fn(
                planner_tokenizer_data, JsonSchemaParser(schema))
            capture_started = perf_counter()
            analysis_record = None
            scope_messages = planner_messages
            if rich_planning:
                # Multi mode: the analysis prompt replaces `instructions` with the
                # frozen ANALYSIS_INSTRUCTIONS, so the multi guidance added in
                # _messages is invisible here; append it explicitly instead.
                # Dispatch-gate scenarios make the delegated values the Planner's
                # OWN next-action inputs, so computing them in the note is policy-
                # compliant (the ban is on pre-solving Worker's unrelated
                # assignments, not on the Planner's own decision inputs).
                multi_extra = ""
                if max_tasks > 1:
                    multi_extra = (
                        " The goal may name several independent values that your "
                        "next-action decision itself depends on; compute each such "
                        "value in this note and show the arithmetic (for example "
                        "'a 120 + b 90 = 210'). In the action reason, state every "
                        "chosen plan branch together with the value that determines "
                        "it. In the action, delegate one bounded verification task "
                        "per value, up to the delegation budget, and make every "
                        "task command a self-contained imperative sentence naming "
                        "the exact value and the public evidence. The command MUST "
                    "quote the exact operand numbers from the record (for "
                    "example 'compute the total as 120 + 90 + 30 = 240'), "
                    "because the Worker sees only the command text and cannot "
                    "look the operands up."
                    )
                # 2026-10-17: sender-side arithmetic scaffold for single-task
                # rich planning; same append-only channel as the multi guidance.
                # 2026-10-18: the instruction is task-count agnostic (it binds
                # every decision value to written-out arithmetic), so multi
                # runs may enable it too; multi already gets a larger budget.
                if arith_scaffold:
                    multi_extra += ARITHMETIC_SCAFFOLD_INSTRUCTION
                analysis_input = analysis_messages(planner_messages, extra_instruction=multi_extra)
                # 2026-10-16: a 320-token note cannot hold K=8 computed values;
                # single-task rich stays at the validated 320 byte-for-byte.
                # 2026-10-17: the arithmetic scaffold needs room for quoted
                # operands and per-step arithmetic, so scaffold runs get 512.
                # 2026-10-18: multi+scaffold may hold up to 8 quoted values,
                # so scaffold multi runs get 1024; scaffold-free multi keeps 768.
                if max_tasks > 1:
                    analysis_max_tokens = 1024 if arith_scaffold else 768
                else:
                    analysis_max_tokens = 512 if arith_scaffold else 320
                analysis = capture_plan(
                    sender, tokenizers["sender"], tokenizers["receiver"],
                    analysis_input, analysis_max_tokens, config["max_input_tokens"],
                    producer_role="Planner rich planning")
                analysis_record = {
                    "output": analysis.output, "prompt_ids": analysis.prompt_ids,
                    "generated_ids": analysis.generated_ids, "prefix_ids": analysis.prefix_ids,
                    "metrics": analysis.metrics,
                    "source_cache_sha256": cache_digest(analysis.source_cache),
                    "source_cache_bytes": cache_bytes(analysis.source_cache),
                }
                scope_messages = analysis_input
                if not analysis.metrics["complete_assistant_turn"] or not analysis.output.strip():
                    stage = {"analysis": analysis_record, "action": None,
                             "wall_seconds": perf_counter() - capture_started,
                             "error": "incomplete_planner_analysis"}
                    stages.append(stage)
                    result["planner_stages"] = stages
                    result["planner"] = stage
                    result.update(status="incomplete_planner_analysis",
                                  wall_seconds=perf_counter() - started)
                    (output / "result.json").write_text(
                        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                    return result
                transcript = capture_plan(
                    sender, tokenizers["sender"], tokenizers["receiver"],
                    ACTION_MESSAGE, action_max_tokens, config["max_input_tokens"],
                    producer_role="Planner rich action", prior_prefix_ids=analysis.prefix_ids,
                    prior_cache=analysis.source_cache, allowed_tokens=allowed)
                transcript.planning_input_ids = analysis.prompt_ids
                transcript.planning_input_text = analysis.prompt_text
                transcript.analysis_token_range = (
                    len(analysis.prompt_ids),
                    len(analysis.prompt_ids) + len(analysis.generated_ids) - 1)
            else:
                transcript = capture_plan(sender, tokenizers["sender"], tokenizers["receiver"],
                                          planner_messages, action_max_tokens, config["max_input_tokens"],
                                          producer_role="Planner core C2C probe", allowed_tokens=allowed)
            action = action_error = None
            try:
                action = json.loads(transcript.output)
            except Exception as exc:
                action_error = f"{type(exc).__name__}: {exc}"
            stage = {
                "output": transcript.output, "prompt_ids": transcript.prompt_ids,
                "generated_ids": transcript.generated_ids, "prefix_ids": transcript.prefix_ids,
                "metrics": transcript.metrics, "action": action, "action_error": action_error,
                "wall_seconds": perf_counter() - capture_started,
                "source_cache_sha256": cache_digest(transcript.source_cache),
                "source_cache_bytes": cache_bytes(transcript.source_cache),
            }
            if analysis_record is not None:
                stage["analysis"] = analysis_record
                stage["analysis_token_range"] = list(transcript.analysis_token_range)
            stages.append(stage)
            if action is None or action.get("action") != "inspect":
                break
        result["planner_stages"] = stages
        result["planner"] = stages[-1]
        if action is None or action.get("action") != "delegate":
            result.update(status="no_actual_delegation", wall_seconds=perf_counter() - started)
            (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            return result
        tasks = action.get("tasks") or []
        if hidden_ablate:
            # A3 receiver-utility ablation: replay the full-selection hidden arm
            # once, then teacher-force its greedy continuation under drop-one
            # selections. Utility of a block = L_full - L_dropone, all under
            # identical published-cache provenance. No gold answer is used:
            # the continuation is the Worker's OWN full-selection output.
            if not rich_planning or transcript.analysis_token_range is None:
                raise ValueError("Ablation requires rich planning with an actual analysis range")
            from .kv import generate_worker, score_worker_continuation
            from .cache_ops import prefill as _prefill
            from .kv import PlanningPayload as _Payload
            lo, hi = transcript.analysis_token_range
            command = _worker_command(tasks[0])
            block_tokens = 16
            blocks = [(b, min(b + block_tokens, hi))
                      for b in range(lo, hi, block_tokens)]
            full_sel = tuple(range(lo, hi))

            def _greedy_and_score(sel):
                spec = _ablation_payload(
                    combined, tokenizers, transcript, receiver, command, sel)
                gen = generate_worker(combined, tokenizers, spec["cmd_ids"],
                                      spec["payload"], "c2c_hidden", 32)
                score = score_worker_continuation(
                    combined, tokenizers, spec["cmd_ids"], spec["payload"],
                    gen["generated_token_ids"])
                return spec, gen, score

            ab_started = perf_counter()
            full_spec, full_gen, full_score = _greedy_and_score(full_sel)
            target_ids = full_gen["generated_token_ids"]
            full_answer = full_gen["output"]
            # 无隐藏上下文基线：receiver 只读短命令
            native_cache, native_seconds = _prefill(receiver, full_spec["cmd_ids"][:-1])
            native_payload = _Payload(tuple(full_spec["cmd_ids"][:-1]), native_cache,
                                      fused=False, hidden_context_tokens=0)
            native_score = score_worker_continuation(
                combined, tokenizers, full_spec["cmd_ids"], native_payload, target_ids)
            block_records = []
            for idx, (blo, bhi) in enumerate(blocks):
                drop_sel = tuple(i for i in full_sel if not (blo <= i < bhi))
                if not drop_sel:
                    continue
                spec, gen, score = _greedy_and_score(drop_sel)
                flipped = int(gen["output"] != full_answer)
                block_records.append({
                    "block_index": idx, "token_range": [blo, bhi],
                    "tokens": bhi - blo,
                    "dropone_sum_logprob": score["sum_logprob"],
                    "dropone_mean_logprob": score["mean_logprob"],
                    "greedy_output": gen["output"],
                    "greedy_flipped_vs_full": flipped,
                    "hidden_context_tokens": len(drop_sel),
                    "wall_seconds": score["wall_seconds"] + gen["service_seconds"],
                    "source_cache_sha256": spec["source_cache_sha256"],
                    "receiver_cache_sha256": spec["receiver_cache_sha256"],
                })
            result["ablation"] = {
                "provenance": "drop-one receiver-utility ablation; continuation is the "
                              "Worker's own full-selection greedy output; no gold answer used",
                "block_tokens": block_tokens,
                "analysis_range": [lo, hi],
                "full_selection": {
                    "sum_logprob": full_score["sum_logprob"],
                    "mean_logprob": full_score["mean_logprob"],
                    "greedy_output": full_answer,
                    "generated_token_ids": target_ids,
                    "hidden_context_tokens": len(full_sel),
                    "receiver_cache_sha256": full_spec["receiver_cache_sha256"],
                },
                "command_native_baseline": {
                    "sum_logprob": native_score["sum_logprob"],
                    "mean_logprob": native_score["mean_logprob"],
                    "prefill_seconds": native_seconds,
                    "hidden_context_tokens": 0,
                },
                "utility_definition": "u_block = full.sum_logprob - dropone.sum_logprob",
                "blocks": block_records,
                "wall_seconds": perf_counter() - ab_started,
            }
            result.update(status="ablation_completed",
                          wall_seconds=perf_counter() - started)
            (output / "result.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            return result
        tasks = action.get("tasks") or []
        if not 1 <= len(tasks) <= max_tasks:
            raise ValueError("Core probe received an invalid actual task count")
        if max_tasks > 1 and len(tasks) < 2:
            # 2026-10-16: the run is no longer aborted -- the Planner's actual
            # delegation is honoured even when it is a single task (observed
            # consistently on the 8-subsystem octo scenarios), the shortfall is
            # recorded, and the Worker arms still run on the real delegation.
            result["planner"]["delegated_fewer_than_ceiling"] = True
            result["planner"]["delegation_shortfall_reason"] = (
                "Planner generated fewer than two tasks under a multi-task ceiling; "
                "actual delegation honoured, arms run on the real tasks")
        selected, scope = state_planning_scope(
            transcript, scope_messages, tokenizers["sender"],
            include_planning=not rich_planning)
        result["planner"]["selected_scope"] = scope
        receiver_prefix = transcript.prefix_ids
        receiver_cache, receiver_seconds = prefill(receiver, receiver_prefix)
        if max_tasks == 1:
            task = tasks[0]
            command = _worker_command(task)
            native, _ = _run_arm(combined, tokenizers, transcript, receiver_prefix,
                                 receiver_cache, command, "text_full")
            # Faithful practical T2T arm: the same Planner continuation produces
            # a useful handoff, which is decoded and then re-encoded for the
            # receiver.  This is the paper's text-to-text comparison, with the
            # actual task command kept identical across all arms.
            handoff_request = json.dumps({
                "invocation_role": "Planner",
                "current_operation": "prepare_worker_handoff",
                "instructions": (
                    "Write a standalone useful handoff for the exact Worker task below. "
                    "Include the relevant public facts, constraints, dependency reasoning, "
                    "evidence and working guidance. Useful calculations are allowed when "
                    "supported. Distinguish missing information and unverified claims. "
                    "Do not emit a scheduler action and do not merely repeat the command."),
                "task": {k: task[k] for k in ("command", "files", "deliverable", "success_criteria")},
            }, ensure_ascii=False)
            handoff_started = perf_counter()
            handoff = capture_plan(
                sender, tokenizers["sender"], tokenizers["receiver"],
                [{"role": "user", "content": handoff_request}], 768,
                config["max_input_tokens"], producer_role="Planner practical T2T handoff",
                prior_prefix_ids=transcript.prefix_ids, prior_cache=transcript.source_cache)
            result["handoff_attempt"] = {
                "output": handoff.output,
                "generated_ids": handoff.generated_ids,
                "metrics": handoff.metrics,
                "request": handoff_request,
            }
            handoff_incomplete = (not handoff.metrics["complete_assistant_turn"]
                                  or not handoff.output.strip())
            if handoff_incomplete:
                # 2026-10-12 repair: an incomplete practical T2T handoff is real
                # evidence about the text channel's reliability; it must be
                # preserved and counted, but it must not abort the other five
                # arms.  text_brief is recorded as failed for this run.
                result["handoff_attempt"]["incomplete_preserved"] = True
                brief = {"arm": "text_brief", "output": "",
                         "arm_error": "incomplete_actual_T2T_handoff",
                         "semantic_status": "arm_failed_preserved",
                         "handoff_incomplete": True}
            else:
                # Put the Worker role boundary in the published T2T prefix.  The
                # command is appended as a separate user turn by worker_ids, so the
                # command remains byte-for-byte the same task contract.
                brief_messages = [
                    {"role": "system", "content": WORKER_ROLE_SYSTEM},
                    {"role": "user", "content": "Planner handoff:\n" + handoff.output},
                ]
                brief_prefix = tokenizers["receiver"].apply_chat_template(
                    brief_messages, tokenize=True, add_generation_prompt=False)
                brief_cache, brief_receiver_seconds = prefill(receiver, brief_prefix)
                brief, _ = _run_arm(
                    combined, tokenizers, transcript, brief_prefix, brief_cache,
                    command, "text_brief", role_system=None, handoff_reencoded=True)
            fused, _ = _run_arm(combined, tokenizers, transcript, receiver_prefix,
                                receiver_cache, command, "c2c_shared", selected)
            if hidden_arms:
                if transcript.analysis_token_range is None:
                    raise ValueError("Hidden arms require the actual rich analysis token range")
                lo, hi = transcript.analysis_token_range
                for name, selection in (
                        ("c2c_hidden_analysis", tuple(range(lo, hi))),
                        ("c2c_hidden_full", selected)):
                    hidden_arm = _run_hidden_arm(
                        combined, tokenizers, transcript, receiver, command,
                        selection, name)
                    result.setdefault("hidden_arms", {})[name] = {
                        "arm": hidden_arm["arm"],
                        "hidden_context_tokens": hidden_arm["hidden_context_tokens"],
                        "hidden_selection_source_indices":
                            hidden_arm["hidden_selection_source_indices"],
                    }
                    result["arms"][name] = hidden_arm
                state_native_arm = _run_state_arm(
                    combined, tokenizers, case, transcript, receiver, command,
                    None, "state_native")
                result["arms"]["state_native"] = state_native_arm
                state_hidden_arm = _run_state_arm(
                    combined, tokenizers, case, transcript, receiver, command,
                    tuple(range(lo, hi)), "c2c_state_hidden")
                result["arms"]["c2c_state_hidden"] = state_hidden_arm
                result.setdefault("hidden_arms", {})["c2c_state_hidden"] = {
                    "arm": "c2c_state_hidden",
                    "hidden_context_tokens": state_hidden_arm["hidden_context_tokens"],
                }
                command_arm = None
                command_arm_error = None
                try:
                    command_arm = _run_command_arm(
                        combined, tokenizers, transcript, receiver, receiver_cache,
                        command, task)
                except Exception as exc:
                    # 2026-10-12 repair: the abandoned c2c_command arm validates
                    # strict string uniqueness; a Planner dispatch whose values
                    # repeat in the texts must not abort the five arms that
                    # already ran.  The arm failure is preserved in the result.
                    command_arm_error = f"{type(exc).__name__}: {exc}"
                if command_arm is not None:
                    result["arms"]["c2c_command"] = command_arm
                    result.setdefault("hidden_arms", {})["c2c_command"] = {
                        "arm": "c2c_command",
                        "command_positions_fused":
                            command_arm["command_positions_fused"],
                        "command_positions_fused_count":
                            command_arm["command_positions_fused_count"],
                    }
                else:
                    result["arms"]["c2c_command"] = {
                        "arm": "c2c_command", "output": "",
                        "arm_error": command_arm_error,
                        "semantic_status": "arm_failed_preserved",
                    }
                    result.setdefault("hidden_arms", {})["c2c_command"] = {
                        "arm": "c2c_command", "arm_error": command_arm_error,
                    }
            result["handoff"] = {
                "request": handoff_request,
                "output": handoff.output,
                "input_ids": handoff.prompt_ids,
                "generated_ids": handoff.generated_ids,
                "metrics": handoff.metrics,
                "wall_seconds": perf_counter() - handoff_started,
                "handoff_is_actual_generated_text": True,
                "handoff_reencoded_for_receiver": True,
            }
            if not handoff_incomplete:
                result["handoff"].update({
                    "receiver_prefix_tokens": len(brief_prefix),
                    "receiver_prefill_seconds": brief_receiver_seconds,
                    "receiver_prefix_sha256": cache_digest(brief_cache),
                    "receiver_prefix_bytes": cache_bytes(brief_cache),
                })
            else:
                result["handoff"]["receiver_prefix_preserved"] = False
            result["arms"].update(
                {"text_full": native, "text_brief": brief, "c2c_shared": fused})
            for arm in result["arms"].values():
                arm["application_gate"] = _os_semantic_gate(case, arm)
                arm["semantic_status"] = (
                    "accepted_by_public_os_gate"
                    if arm["application_gate"]["accepted"] else
                    "rejected_by_public_os_gate")
        else:
            from .kv import fuse_context
            fusion_timings = {}
            fused_cache, fusion_seconds = fuse_context(
                combined, transcript.source_cache, receiver_cache,
                token_indices=selected, timings=fusion_timings)
            native_calls, brief_calls, c2c_calls = [], [], []
            hidden_call_lists = {}
            for task in tasks:
                command = _worker_command(task)
                native, _ = _run_arm(combined, tokenizers, transcript, receiver_prefix,
                                     receiver_cache, command, "text_full")
                # Faithful per-task T2T arm: each Worker task gets its own actual
                # generated handoff, decoded and re-encoded for the receiver.
                # This is the repeated strong-model handoff cost that one shared
                # C2C fusion replaces.
                handoff_request = json.dumps({
                    "invocation_role": "Planner",
                    "current_operation": "prepare_worker_handoff",
                    "instructions": (
                        "Write a standalone useful handoff for the exact Worker task below. "
                        "Include the relevant public facts, constraints, dependency reasoning, "
                        "evidence and working guidance. Useful calculations are allowed when "
                        "supported. Distinguish missing information and unverified claims. "
                        "Do not emit a scheduler action and do not merely repeat the command."),
                    "task": {k: task[k] for k in ("command", "files", "deliverable", "success_criteria")},
                }, ensure_ascii=False)
                handoff_started = perf_counter()
                handoff = capture_plan(
                    sender, tokenizers["sender"], tokenizers["receiver"],
                    [{"role": "user", "content": handoff_request}], 768,
                    config["max_input_tokens"], producer_role="Planner practical T2T handoff",
                    prior_prefix_ids=transcript.prefix_ids, prior_cache=transcript.source_cache)
                handoff_incomplete = (not handoff.metrics["complete_assistant_turn"]
                                      or not handoff.output.strip())
                if handoff_incomplete:
                    # 2026-10-14 repair: same policy as the single-task branch —
                    # an incomplete practical T2T handoff is real evidence about
                    # the text channel's reliability; it is preserved and counted,
                    # and it must not abort the other arms or the remaining tasks.
                    result.setdefault("failed_handoff_attempts", []).append({
                        "task_command": task["command"],
                        "request": handoff_request,
                        "output": handoff.output,
                        "generated_ids_tail": handoff.generated_ids[-10:],
                        "metrics": handoff.metrics,
                    })
                    brief = {"arm": "text_brief", "output": "",
                             "arm_error": "incomplete_actual_T2T_handoff",
                             "semantic_status": "arm_failed_preserved",
                             "handoff_incomplete": True,
                             "handoff": {
                                 "output": handoff.output,
                                 "metrics": handoff.metrics,
                                 "wall_seconds": perf_counter() - handoff_started,
                             }}
                else:
                    brief_messages = [
                        {"role": "system", "content": WORKER_ROLE_SYSTEM},
                        {"role": "user", "content": "Planner handoff:\n" + handoff.output},
                    ]
                    brief_prefix = tokenizers["receiver"].apply_chat_template(
                        brief_messages, tokenize=True, add_generation_prompt=False)
                    brief_cache, brief_receiver_seconds = prefill(receiver, brief_prefix)
                    brief, _ = _run_arm(
                        combined, tokenizers, transcript, brief_prefix, brief_cache,
                        command, "text_brief", role_system=None, handoff_reencoded=True)
                    brief["handoff"] = {
                        "output": handoff.output,
                        "metrics": handoff.metrics,
                        "wall_seconds": perf_counter() - handoff_started,
                        "receiver_prefill_seconds": brief_receiver_seconds,
                    }
                fused, _ = _run_arm(combined, tokenizers, transcript, receiver_prefix,
                                    receiver_cache, command, "c2c_shared", selected,
                                    prepared_cache=fused_cache,
                                    prepared_fusion_seconds=fusion_seconds,
                                    prepared_fusion_timings=fusion_timings)
                native_calls.append(native)
                brief_calls.append(brief)
                c2c_calls.append(fused)
                if hidden_arms:
                    # 2026-10-21: hidden arms (probe4/5 lineage) extended to the
                    # multi-task branch so a one-shot N-task delegation can be
                    # compared under the same arms as the single-task probes.
                    if transcript.analysis_token_range is None:
                        raise ValueError("Hidden arms require the actual rich analysis token range")
                    lo, hi = transcript.analysis_token_range
                    for name, selection in (
                            ("c2c_hidden_analysis", tuple(range(lo, hi))),
                            ("c2c_hidden_full", selected)):
                        hidden_arm = _run_hidden_arm(
                            combined, tokenizers, transcript, receiver, command,
                            selection, name)
                        hidden_call_lists.setdefault(name, []).append(hidden_arm)
                    state_native_arm = _run_state_arm(
                        combined, tokenizers, case, transcript, receiver, command,
                        None, "state_native")
                    hidden_call_lists.setdefault("state_native", []).append(state_native_arm)
                    state_hidden_arm = _run_state_arm(
                        combined, tokenizers, case, transcript, receiver, command,
                        tuple(range(lo, hi)), "c2c_state_hidden")
                    hidden_call_lists.setdefault("c2c_state_hidden", []).append(state_hidden_arm)
                    try:
                        command_arm = _run_command_arm(
                            combined, tokenizers, transcript, receiver, receiver_cache,
                            command, task)
                    except Exception as exc:
                        # same repair policy as the single-task branch: a failed
                        # c2c_command validation must not abort the other arms.
                        command_arm = {"arm": "c2c_command", "output": "",
                                       "arm_error": f"{type(exc).__name__}: {exc}",
                                       "semantic_status": "arm_failed_preserved"}
                    hidden_call_lists.setdefault("c2c_command", []).append(command_arm)
            result["arms"] = {
                "text_full": {"calls": native_calls, "shared_receiver_prefill": True},
                "text_brief": {"calls": brief_calls,
                               "handoff_generated_per_task": True,
                               "handoff_is_actual_generated_text": True,
                               "handoff_reencoded_for_receiver": True},
                "c2c_shared": {"calls": c2c_calls, "shared_fused_cache": True,
                               "fusion_seconds": fusion_seconds, "fusion_timings": fusion_timings},
            }
            for name, calls in hidden_call_lists.items():
                result["arms"][name] = {"calls": calls}
                result.setdefault("hidden_arms", {})[name] = {
                    "arm": name, "call_count": len(calls),
                    "hidden_context_tokens": calls[0].get("hidden_context_tokens"),
                }
        result["preparation"] = {
            "receiver_prefill_seconds": receiver_seconds,
            "selected_indices": list(selected), "selected_count": len(selected),
            "source_cache_sha256": cache_digest(transcript.source_cache),
            "receiver_native_cache_sha256": cache_digest(receiver_cache),
            "receiver_native_cache_bytes": cache_bytes(receiver_cache),
            "kv_bytes_are_not_network_traffic": True,
            "physical_zero_copy": False,
        }
        result["application_contract"] = {
            "online_gate": "public_state_derived_worker_report_check",
            "offline_expected_answers_read": False,
            "planner_dependency_contract_valid": all(
                task_state == "SUCCEEDED"
                for task in tasks
                for task_state in [
                    next((item["state"] for item in case["public_background"]["tasks"]
                          if item["id"] == dependency), None)
                    for dependency in task.get("depends_on", [])
                ]
            ),
            "parent_acceptance_requires_worker_gate": True,
        }
        if feedback_loop and max_tasks == 1:
            feedback = {}
            task = tasks[0]
            for method, arm in result["arms"].items():
                worker_output = arm.get("output", "")
                observation = {
                    "worker_output": worker_output,
                    "worker_output_sha256": hashlib.sha256(
                        worker_output.encode("utf-8")).hexdigest(),
                    "application_gate": arm.get("application_gate"),
                    "task_command": task.get("command"),
                    "context_version": "planner-dispatch-1",
                    "public_state_sha256": result["public_sha256"],
                    "arm_method": method,
                    "source_cache_sha256": arm.get("source_cache_sha256"),
                    "receiver_cache_sha256": arm.get("receiver_cache_sha256"),
                    "receiver_cache_bytes": arm.get("receiver_cache_bytes"),
                    "worker_consumed_fused_cache": arm.get("worker_consumed_fused_cache"),
                    "source_cache_was_reencoded": arm.get("source_cache_was_reencoded"),
                    "observation_origin": "actual Worker terminal output and public-state gate",
                }
                feedback_messages = _messages(
                    case, max_tasks=1, last_decision="delegate",
                    observation={"worker_observation": observation}, remaining_decisions=1)
                feedback_started = perf_counter()
                feedback_analysis = None
                try:
                    feedback_vocab = build_token_enforcer_tokenizer_data(tokenizers["sender"])
                    feedback_allowed = build_transformers_prefix_allowed_tokens_fn(
                        feedback_vocab, JsonSchemaParser(schema))
                    if rich_planning:
                        feedback_input = analysis_messages(
                            feedback_messages,
                            extra_instruction=FEEDBACK_REVIEW_INSTRUCTION)
                        feedback_analysis = capture_plan(
                            sender, tokenizers["sender"], tokenizers["receiver"],
                            feedback_input, 320, config["max_input_tokens"],
                            producer_role="Planner feedback analysis")
                        if not feedback_analysis.metrics["complete_assistant_turn"]:
                            raise ValueError("incomplete_feedback_analysis")
                        feedback_transcript = capture_plan(
                            sender, tokenizers["sender"], tokenizers["receiver"],
                            ACTION_MESSAGE, 256, config["max_input_tokens"],
                            producer_role="Planner feedback action",
                            prior_prefix_ids=feedback_analysis.prefix_ids,
                            prior_cache=feedback_analysis.source_cache,
                            allowed_tokens=feedback_allowed)
                    else:
                        feedback_transcript = capture_plan(
                            sender, tokenizers["sender"], tokenizers["receiver"],
                            feedback_messages, 256, config["max_input_tokens"],
                            producer_role="Planner feedback action", allowed_tokens=feedback_allowed)
                    feedback_action = json.loads(feedback_transcript.output)
                    consumption = _feedback_consumption_evidence(
                        observation,
                        feedback_analysis.output if feedback_analysis is not None else "",
                        feedback_transcript.output,
                    )
                    parent_gate = _feedback_parent_gate(case, feedback_action, observation)
                    feedback[method] = {
                        "observation": observation,
                        "analysis": ({
                            "output": feedback_analysis.output,
                            "generated_ids": feedback_analysis.generated_ids,
                            "metrics": feedback_analysis.metrics,
                            "prompt_ids": feedback_analysis.prompt_ids,
                            "prefix_ids": feedback_analysis.prefix_ids,
                            "source_cache_sha256": cache_digest(feedback_analysis.source_cache),
                            "source_cache_bytes": cache_bytes(feedback_analysis.source_cache),
                        } if feedback_analysis is not None else None),
                        "output": feedback_transcript.output,
                        "action": feedback_action,
                        "prompt_ids": feedback_transcript.prompt_ids,
                        "generated_ids": feedback_transcript.generated_ids,
                        "prefix_ids": feedback_transcript.prefix_ids,
                        "source_cache_sha256": cache_digest(feedback_transcript.source_cache),
                        "wall_seconds": perf_counter() - feedback_started,
                        "consumed_worker_observation": consumption["consumed"],
                        "consumption_evidence": consumption,
                        "parent_acceptance_gate": parent_gate,
                    }
                except Exception as exc:
                    feedback[method] = {
                        "observation": observation,
                        "error": f"{type(exc).__name__}: {exc}",
                        "wall_seconds": perf_counter() - feedback_started,
                        "consumed_worker_observation": False,
                        "consumption_evidence": {"consumed": False, "markers": []},
                        "parent_acceptance_gate": {"accepted": False, "reason": "feedback generation failed", "host_generated_repair": False},
                    }
            result["feedback_planner"] = feedback
        if max_tasks == 1:
            result["preparation"].update({
                "text_brief_extra_generation_seconds": result["handoff"]["metrics"]["prefill_seconds"] + result["handoff"]["metrics"]["generation_seconds"] + result["handoff"]["metrics"]["close_turn_seconds"],
                "text_brief_receiver_prefill_seconds": result["handoff"].get("receiver_prefill_seconds"),
                "text_brief_generated_tokens": result["handoff"]["metrics"]["generated_tokens"],
                "c2c_fusion_seconds": fused.get("fusion_seconds", 0.0),
                "cost_scope_note": "Planner handoff generation, receiver prefill, fusion and Worker service are recorded separately; KV bytes are not network speed.",
            })
        if max_tasks > 1:
            result["preparation"].update({
                "shared_fused_cache_sha256": cache_digest(fused_cache),
                "shared_fused_cache_bytes": cache_bytes(fused_cache),
                "shared_fusion_seconds": fusion_seconds,
                "shared_fusion_reused_by_each_worker": True,
                "worker_cache_branches_are_private": True,
            })
        result.update(status="completed", wall_seconds=perf_counter() - started)
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}", wall_seconds=perf_counter() - started)
        raise
    finally:
        (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run_c2c_core_acb(output_dir, *, public_filename="public.json", max_tasks=8,
                     arith_scaffold=True):
    """B1: ACB-scheduled multi-task C2C run on one Planner dispatch.

    The Planner rich-plans once; every delegated task becomes an Agent Control
    Block scheduled by agent_runtime.acb (states, DAG dependencies, GPU-budget
    admission accounting, heartbeat liveness, bounded repairs). Each task
    executes through the real hidden-arm primitive (Worker reads only the
    short command; the Planner analysis KV is fused as hidden context) and is
    accepted only through the public OS semantic gate. Design and honest
    limits: docs/b1-acb-design-2026-10-18.md.
    """
    import torch
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import (
        build_token_enforcer_tokenizer_data,
        build_transformers_prefix_allowed_tokens_fn,
    )
    from .loading import load_models
    from .cache_ops import cache_digest, cache_bytes
    from .planning_policy import (ACTION_MESSAGE, analysis_messages,
                                  ARITHMETIC_SCAFFOLD_INSTRUCTION)
    from .transcript import capture_plan
    from .acb import AgentControlBlock, ACBScheduler

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    public_path = ROOT / "data/c2c-core-os" / public_filename
    case = json.loads(public_path.read_text(encoding="utf-8"))
    config = json.loads((ROOT / "configs/cloud.json").read_text())
    lock = json.loads((ROOT / "configs/cloud.models.lock.json").read_text())
    result = {
        "status": "running",
        "scope": "ACB-scheduled multi-task C2C: one Planner rich dispatch, tasks "
                 "scheduled as Agent Control Blocks (DAG deps, GPU-budget "
                 "admission, heartbeat, bounded repairs), each executed as the "
                 "hidden-analysis C2C arm through the public OS gate",
        "planner_rich_planning": True, "planner_arith_scaffold": arith_scaffold,
        "planner_feedback_loop": False,
        "public_sha256": _sha(public_path), "config": config, "model_pins": lock,
        "source_sha256": {
            public_filename: _sha(public_path),
            "c2c_core.py": _sha(ROOT / "src/agent_runtime/c2c_core.py"),
            "scripts/run.py": _sha(ROOT / "scripts/run.py"),
            "src/agent_runtime/acb.py": _sha(ROOT / "src/agent_runtime/acb.py"),
            "src/agent_runtime/kv.py": _sha(ROOT / "src/agent_runtime/kv.py"),
            "src/agent_runtime/transcript.py": _sha(ROOT / "src/agent_runtime/transcript.py"),
            "src/agent_runtime/planning_policy.py": _sha(ROOT / "src/agent_runtime/planning_policy.py"),
            "src/agent_runtime/cache_ops.py": _sha(ROOT / "src/agent_runtime/cache_ops.py"),
        },
        "pid": os.getpid(), "model_service_separate_process": False,
        "case_id": case["id"], "planner": None, "planner_stages": [],
        "acb_report": None, "scheduler_events": [],
        "semantic_success": False,
        "semantic_status": "requires independent offline audit of raw Worker outputs",
    }
    started = perf_counter()
    try:
        combined, tokenizers = load_models(config, lock)
        combined.multi_source_fusion_mode = "sequential"
        sender, receiver = combined.model_list[1], combined.model_list[0]
        schema = _core_schema(max_tasks=max_tasks)
        action_max_tokens = 128 * max_tasks

        # --- Planner rich planning stage (same provenance as the multi path) ---
        planner_messages = _messages(case, max_tasks=max_tasks, remaining_decisions=2)
        planner_tokenizer_data = build_token_enforcer_tokenizer_data(tokenizers["sender"])
        allowed = build_transformers_prefix_allowed_tokens_fn(
            planner_tokenizer_data, JsonSchemaParser(schema))
        capture_started = perf_counter()
        multi_extra = (
            " The goal may name several independent values that your "
            "next-action decision itself depends on; compute each such "
            "value in this note and show the arithmetic (for example "
            "'a 120 + b 90 = 210'). In the action reason, state every "
            "chosen plan branch together with the value that determines "
            "it. In the action, delegate one bounded verification task "
            "per value, up to the delegation budget, and make every "
            "task command a self-contained imperative sentence naming "
            "the exact value and the public evidence."
        )
        if arith_scaffold:
            multi_extra += ARITHMETIC_SCAFFOLD_INSTRUCTION
        analysis_input = analysis_messages(planner_messages, extra_instruction=multi_extra)
        analysis_max_tokens = 1024 if max_tasks > 1 else 512
        analysis = capture_plan(
            sender, tokenizers["sender"], tokenizers["receiver"],
            analysis_input, analysis_max_tokens, config["max_input_tokens"],
            producer_role="Planner rich planning")
        analysis_record = {
            "output": analysis.output, "prompt_ids": analysis.prompt_ids,
            "generated_ids": analysis.generated_ids, "prefix_ids": analysis.prefix_ids,
            "metrics": analysis.metrics,
            "source_cache_sha256": cache_digest(analysis.source_cache),
            "source_cache_bytes": cache_bytes(analysis.source_cache),
        }
        if not analysis.metrics["complete_assistant_turn"] or not analysis.output.strip():
            result.update(status="incomplete_planner_analysis",
                          wall_seconds=perf_counter() - started)
            return result
        transcript = capture_plan(
            sender, tokenizers["sender"], tokenizers["receiver"],
            ACTION_MESSAGE, action_max_tokens, config["max_input_tokens"],
            producer_role="Planner rich action", prior_prefix_ids=analysis.prefix_ids,
            prior_cache=analysis.source_cache, allowed_tokens=allowed)
        transcript.planning_input_ids = analysis.prompt_ids
        transcript.planning_input_text = analysis.prompt_text
        transcript.analysis_token_range = (
            len(analysis.prompt_ids),
            len(analysis.prompt_ids) + len(analysis.generated_ids) - 1)
        action = action_error = None
        try:
            action = json.loads(transcript.output)
        except Exception as exc:
            action_error = f"{type(exc).__name__}: {exc}"
        stage = {
            "output": transcript.output, "prompt_ids": transcript.prompt_ids,
            "generated_ids": transcript.generated_ids, "prefix_ids": transcript.prefix_ids,
            "metrics": transcript.metrics, "action": action, "action_error": action_error,
            "wall_seconds": perf_counter() - capture_started,
            "source_cache_sha256": cache_digest(transcript.source_cache),
            "source_cache_bytes": cache_bytes(transcript.source_cache),
            "analysis": analysis_record,
            "analysis_token_range": list(transcript.analysis_token_range),
        }
        result["planner_stages"] = [stage]
        result["planner"] = stage
        if action is None or action.get("action") != "delegate":
            result.update(status="no_actual_delegation",
                          wall_seconds=perf_counter() - started)
            if action is not None and (case.get("probe") or {}).get("family") in OS_CLASSICS_FAMILIES:
                result["planner_finish_audit"] = _audit_planner_finish(case, action)
            return result

        tasks = action.get("tasks") or []
        tasks = _apply_declared_workflow(action, tasks, result)
        lo, hi = transcript.analysis_token_range
        selection = tuple(range(lo, hi))
        if not selection:
            result.update(status="no_analysis_selection",
                          wall_seconds=perf_counter() - started)
            return result

        # --- ACB backend: hidden-arm execution + public gate, wait accounting ---
        free_b, _total_b = torch.cuda.mem_get_info()
        gpu_budget = int(free_b) - 4 * 1024 ** 3  # 4 GiB safety reserve
        est_bytes = 200 * 1024 ** 2               # ~measured receiver_cache_bytes order

        class HiddenBackend:
            gpu_budget_bytes = max(gpu_budget, 0)

            def heartbeat(self, acb):
                # Serial execution: liveness is confirmed pre-flight per task;
                # a returned timestamp means the runtime process is alive.
                return perf_counter()

            def execute(self, acb):
                try:
                    arm = _run_hidden_arm(
                        combined, tokenizers, transcript, receiver,
                        acb.command, selection, f"acb_{acb.task_id}")
                except Exception as exc:
                    return {"artifact": None, "gate": None,
                            "error": f"{type(exc).__name__}: {exc}",
                            "wait_states": {"WAIT_MODEL": 0.0}}
                gate = _os_semantic_gate(case, arm, acb.command)
                arm["application_gate"] = gate
                arm["semantic_status"] = (
                    "accepted_by_public_os_gate" if gate["accepted"]
                    else "rejected_by_public_os_gate")
                arms_dir = output / "tasks"
                arms_dir.mkdir(exist_ok=True)
                (arms_dir / f"{acb.task_id}.json").write_text(
                    json.dumps(arm, ensure_ascii=False, indent=1), encoding="utf-8")
                wait_model = float(arm.get("service_seconds") or 0.0)
                wait_fusion = float(arm.get("fusion_seconds") or 0.0) \
                    + float(arm.get("reference_prefill_seconds") or 0.0)
                return {"artifact": arm.get("output"), "gate": gate, "error": None,
                        "record": arm,
                        "wait_states": {"WAIT_MODEL": wait_model,
                                        "WAIT_FUSION": wait_fusion}}

        def repair_policy(acb):
            reason = (acb.gate or {}).get("reason") or acb.error or "unknown"
            return (f"{acb.command}\n[runtime feedback] A previous attempt of this "
                    f"task was rejected ({reason}). Recompute the value from the "
                    f"public evidence and report only the corrected result.")

        events = []
        scheduler = ACBScheduler(HiddenBackend(), repair_policy=repair_policy,
                                 emit=events.append)

        def _map_deps(raw_deps, index):
            mapped = []
            for d in raw_deps or []:
                d = str(d).strip().lower()
                if d.startswith("t") and d[1:].isdigit() \
                        and 1 <= int(d[1:]) <= len(tasks) and int(d[1:]) != index + 1:
                    mapped.append(f"t{int(d[1:])}")
            return tuple(mapped)

        for i, task in enumerate(tasks):
            acb = AgentControlBlock(
                task_id=f"t{i + 1}",
                command=_worker_command(task),
                deliverable=str(task.get("deliverable", "")),
                depends_on=_map_deps(task.get("depends_on"), i),
                priority=float(len(tasks) - i),
                gpu_bytes_est=est_bytes,
                est_tokens=int(analysis.metrics.get("generated_tokens") or 0),
                max_attempts=2)
            scheduler.submit(acb)

        report = scheduler.run()
        result["acb_report"] = report
        result["scheduler_events"] = events
        result["arms"] = {"c2c_hidden_analysis": {
            "calls": list(report["tasks"].values()),
            "note": "each ACB task executed as the hidden-analysis C2C arm; "
                    "per-task raw arm records under tasks/<id>.json"}}
        result.update(status="completed",
                      wall_seconds=perf_counter() - started)
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      wall_seconds=perf_counter() - started)
        raise
    finally:
        (output / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run_c2c_core_acb_isolated(output_dir, *, public_filename="public.json", max_tasks=8,
                              arith_scaffold=True, worker_memory_mib=24576,
                              worker_timeout_s=300, sched_policy="priority",
                              sim_quota=None, payload_mmap=False,
                              payload_pool_capacity=2):
    """B1 isolated variant: every Worker runs as a real OS child process.

    Identical Planner provenance and ACB scheduling to run_c2c_core_acb; the
    execution seam changes (PPT slide 8: Agent = OS subprocess + cgroup
    limits). The kernel (this process) still owns the Planner, the fuser and
    GPU fusion: it publishes each task's fused receiver cache plus command
    metadata to files (cache payload built by the verified _ablation_payload
    seam), then spawns ``python -m agent_runtime.worker_entry`` through
    agent_runtime.os_isolate with memory/pids/CPU limits and a wall-clock
    leash. Child crash, OOM kill or hang is contained: the parent survives,
    the task becomes FAILED, and the bounded repair loop re-submits it with
    the gate's rejection reason. Per-task cache/meta/record files stay under
    tasks/ as execution evidence.

    sched_policy: "priority" (baseline ACBScheduler) or "fair"
    (os_sched.FairScheduler, CFS min-vruntime ordering + wait-for deadlock
    probe; admission "fit").
    sim_quota: optional dict {"tokens": int, "wall_s": float,
    "memory_mib": float, "slice_pids_max": int} — enables the simulated
    cgroup v2 controller (metering + hard quota kills); None disables it.
    payload_mmap: publish the receiver cache as a raw contiguous bin that the
    child maps with np.memmap (host-level zero copy: shared page cache,
    no deserialization copy); False keeps torch.load.  GPU device zero-copy
    is not implemented in either mode (.to(cuda) copies).
    payload_pool_capacity: LRU pool of resident published cache payloads;
    publishing beyond capacity swap-outs the least-recently-used payload to
    the swap area, and a re-execution that finds its payload swapped out
    records a page fault plus real swap-in cost.
    """
    import os
    import sys
    import torch
    from huggingface_hub import snapshot_download
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import (
        build_token_enforcer_tokenizer_data,
        build_transformers_prefix_allowed_tokens_fn,
    )
    from .loading import load_models
    from .cache_ops import cache_digest, cache_bytes
    from .planning_policy import (ACTION_MESSAGE, analysis_messages,
                                  ARITHMETIC_SCAFFOLD_INSTRUCTION)
    from .transcript import capture_plan
    from .acb import AgentControlBlock, ACBScheduler
    from .os_isolate import IsolationLimits, run_isolated

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    public_path = ROOT / "data/c2c-core-os" / public_filename
    case = json.loads(public_path.read_text(encoding="utf-8"))
    config = json.loads((ROOT / "configs/cloud.json").read_text())
    lock = json.loads((ROOT / "configs/cloud.models.lock.json").read_text())
    receiver_path = snapshot_download(
        **lock["receiver"], cache_dir=ROOT / ".cache/huggingface/hub",
        local_files_only=True)
    result = {
        "status": "running",
        "scope": "ACB-scheduled multi-task C2C, ISOLATED workers: one Planner "
                 "rich dispatch; each task scheduled as an Agent Control Block "
                 "and executed in a real OS child process (cgroup/rlimit "
                 "limits, wall-clock leash) that consumes a published fused "
                 "KV payload; public OS gate accepts every artifact; bounded "
                 "repairs re-enter the scheduler",
        "planner_rich_planning": True, "planner_arith_scaffold": arith_scaffold,
        "planner_feedback_loop": False,
        "worker_execution": "isolated_os_subprocess",
        "worker_limits": {"memory_mib": worker_memory_mib,
                          "cpu_quota_pct": 100, "pids_max": 128,
                          "timeout_s": worker_timeout_s},
        "public_sha256": _sha(public_path), "config": config, "model_pins": lock,
        "source_sha256": {
            public_filename: _sha(public_path),
            "c2c_core.py": _sha(ROOT / "src/agent_runtime/c2c_core.py"),
            "scripts/run.py": _sha(ROOT / "scripts/run.py"),
            "src/agent_runtime/acb.py": _sha(ROOT / "src/agent_runtime/acb.py"),
            "src/agent_runtime/os_isolate.py": _sha(ROOT / "src/agent_runtime/os_isolate.py"),
            "src/agent_runtime/worker_entry.py": _sha(ROOT / "src/agent_runtime/worker_entry.py"),
            "src/agent_runtime/kv.py": _sha(ROOT / "src/agent_runtime/kv.py"),
            "src/agent_runtime/transcript.py": _sha(ROOT / "src/agent_runtime/transcript.py"),
            "src/agent_runtime/planning_policy.py": _sha(ROOT / "src/agent_runtime/planning_policy.py"),
            "src/agent_runtime/cache_ops.py": _sha(ROOT / "src/agent_runtime/cache_ops.py"),
        },
        "pid": os.getpid(), "model_service_separate_process": False,
        "worker_processes_are_os_children": True,
        "case_id": case["id"], "planner": None, "planner_stages": [],
        "acb_report": None, "scheduler_events": [],
        "semantic_success": False,
        "semantic_status": "requires independent offline audit of raw Worker outputs",
    }
    started = perf_counter()
    try:
        combined, tokenizers = load_models(config, lock)
        combined.multi_source_fusion_mode = "sequential"
        sender, receiver = combined.model_list[1], combined.model_list[0]
        schema = _core_schema(max_tasks=max_tasks)
        action_max_tokens = 128 * max_tasks

        # --- Planner rich planning stage: same provenance as the multi path ---
        planner_messages = _messages(case, max_tasks=max_tasks, remaining_decisions=2)
        planner_tokenizer_data = build_token_enforcer_tokenizer_data(tokenizers["sender"])
        allowed = build_transformers_prefix_allowed_tokens_fn(
            planner_tokenizer_data, JsonSchemaParser(schema))
        capture_started = perf_counter()
        multi_extra = (
            " The goal may name several independent values that your "
            "next-action decision itself depends on; compute each such "
            "value in this note and show the arithmetic (for example "
            "'a 120 + b 90 = 210'). In the action reason, state every "
            "chosen plan branch together with the value that determines "
            "it. In the action, delegate one bounded verification task "
            "per value, up to the delegation budget, and make every "
            "task command a self-contained imperative sentence naming "
            "the exact value and the public evidence."
        )
        if arith_scaffold:
            multi_extra += ARITHMETIC_SCAFFOLD_INSTRUCTION
        analysis_input = analysis_messages(planner_messages, extra_instruction=multi_extra)
        analysis_max_tokens = 1024 if max_tasks > 1 else 512
        analysis = capture_plan(
            sender, tokenizers["sender"], tokenizers["receiver"],
            analysis_input, analysis_max_tokens, config["max_input_tokens"],
            producer_role="Planner rich planning")
        analysis_record = {
            "output": analysis.output, "prompt_ids": analysis.prompt_ids,
            "generated_ids": analysis.generated_ids, "prefix_ids": analysis.prefix_ids,
            "metrics": analysis.metrics,
            "source_cache_sha256": cache_digest(analysis.source_cache),
            "source_cache_bytes": cache_bytes(analysis.source_cache),
        }
        if not analysis.metrics["complete_assistant_turn"] or not analysis.output.strip():
            result.update(status="incomplete_planner_analysis",
                          wall_seconds=perf_counter() - started)
            return result
        transcript = capture_plan(
            sender, tokenizers["sender"], tokenizers["receiver"],
            ACTION_MESSAGE, action_max_tokens, config["max_input_tokens"],
            producer_role="Planner rich action", prior_prefix_ids=analysis.prefix_ids,
            prior_cache=analysis.source_cache, allowed_tokens=allowed)
        transcript.planning_input_ids = analysis.prompt_ids
        transcript.planning_input_text = analysis.prompt_text
        transcript.analysis_token_range = (
            len(analysis.prompt_ids),
            len(analysis.prompt_ids) + len(analysis.generated_ids) - 1)
        action = action_error = None
        try:
            action = json.loads(transcript.output)
        except Exception as exc:
            action_error = f"{type(exc).__name__}: {exc}"
        stage = {
            "output": transcript.output, "prompt_ids": transcript.prompt_ids,
            "generated_ids": transcript.generated_ids, "prefix_ids": transcript.prefix_ids,
            "metrics": transcript.metrics, "action": action, "action_error": action_error,
            "wall_seconds": perf_counter() - capture_started,
            "source_cache_sha256": cache_digest(transcript.source_cache),
            "source_cache_bytes": cache_bytes(transcript.source_cache),
            "analysis": analysis_record,
            "analysis_token_range": list(transcript.analysis_token_range),
        }
        result["planner_stages"] = [stage]
        result["planner"] = stage
        if action is None or action.get("action") != "delegate":
            result.update(status="no_actual_delegation",
                          wall_seconds=perf_counter() - started)
            if action is not None and (case.get("probe") or {}).get("family") in OS_CLASSICS_FAMILIES:
                result["planner_finish_audit"] = _audit_planner_finish(case, action)
            return result

        tasks = action.get("tasks") or []
        tasks = _apply_declared_workflow(action, tasks, result)
        lo, hi = transcript.analysis_token_range
        selection = tuple(range(lo, hi))
        if not selection:
            result.update(status="no_analysis_selection",
                          wall_seconds=perf_counter() - started)
            return result

        # --- Isolated backend: kernel publishes fused payload, child consumes ---
        free_b, _total_b = torch.cuda.mem_get_info()
        gpu_budget = int(free_b) - 4 * 1024 ** 3  # 4 GiB safety reserve
        est_bytes = 200 * 1024 ** 2               # ~measured receiver_cache_bytes order
        arms_dir = output / "tasks"
        arms_dir.mkdir(exist_ok=True)
        child_env = dict(os.environ)
        child_env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep \
            + child_env.get("PYTHONPATH", "")

        def repair_policy(acb):
            reason = (acb.gate or {}).get("reason") or acb.error or "unknown"
            return (f"{acb.command}\n[runtime feedback] A previous attempt of this "
                    f"task was rejected ({reason}). Recompute the value from the "
                    f"public evidence and report only the corrected result.")

        # Events are kept in a closure list (not on the backend instance) so
        # the record survives ControlledBackend decoration.
        pool_events = []

        class IsolatedBackend:
            gpu_budget_bytes = max(gpu_budget, 0)

            def __init__(self):
                # LRU pool of resident published payloads (demand-paging
                # simulation): insertion order is least- to most-recently
                # used; evictions move the file to the swap area and a
                # re-execution swap-in is accounted as a page fault.
                self.payload_pool = {}
                self.swap_dir = output / "swap"
                self.swap_dir.mkdir(exist_ok=True)

            def _pool_touch(self, acb, cache_file):
                """Evict LRU payloads until the pool has room, then swap-in.

                A page fault fires when the payload this execution depends
                on (its own earlier attempt or its base task's payload)
                was evicted to the swap area and must be demand-paged back.
                """
                page_fault = False
                swap_in_seconds = 0.0
                base = acb.task_id.split(".")[0]
                base_file = cache_file.with_name(
                    cache_file.name.replace(acb.task_id, base, 1))
                candidates = [cache_file, base_file]
                for wanted in candidates:
                    swapped = self.swap_dir / wanted.name
                    if not wanted.exists() and swapped.exists():
                        fault_started = perf_counter()
                        swapped.rename(wanted)
                        swap_in_seconds = perf_counter() - fault_started
                        page_fault = True
                        pool_events.append({
                            "event": "payload_page_fault",
                            "task_id": acb.task_id,
                            "swapped_in": wanted.name,
                            "swap_in_seconds": swap_in_seconds})
                        break
                while len(self.payload_pool) >= payload_pool_capacity:
                    evict_id, evict_path = next(iter(self.payload_pool.items()))
                    del self.payload_pool[evict_id]
                    if Path(evict_path).exists():
                        Path(evict_path).rename(self.swap_dir / Path(evict_path).name)
                    pool_events.append({"event": "payload_swap_out",
                                        "task_id": evict_id})
                self.payload_pool[acb.task_id] = str(cache_file)
                return page_fault, swap_in_seconds

            def heartbeat(self, acb):
                # The child PID is recorded per attempt; liveness is confirmed
                # pre-flight and the wall-clock leash bounds any hang.
                return perf_counter()

            def execute(self, acb):
                try:
                    payload_started = perf_counter()
                    payload_info = _ablation_payload(
                        combined, tokenizers, transcript, receiver,
                        acb.command, selection, case=case)
                    payload = payload_info["payload"]
                    cmd_ids = payload_info["cmd_ids"]
                    suffix = ".cache.bin" if payload_mmap else ".cache.pt"
                    cache_file = arms_dir / f"{acb.task_id}{suffix}"
                    meta_file = arms_dir / f"{acb.task_id}.meta.json"
                    out_file = arms_dir / f"{acb.task_id}.out.json"
                    if payload_mmap:
                        layout = []
                        with open(cache_file, "wb") as fh:
                            for index, (key, value) in enumerate(payload.receiver_cache):
                                k = key.detach().cpu().contiguous()
                                v = value.detach().cpu().contiguous()
                                fh.write(k.view(torch.uint8).numpy().tobytes())
                                fh.write(v.view(torch.uint8).numpy().tobytes())
                                layout.append({"index": index,
                                               "k_shape": list(k.shape),
                                               "v_shape": list(v.shape)})
                    else:
                        layout = None
                        torch.save(
                            {index: (key.detach().cpu(), value.detach().cpu())
                             for index, (key, value) in enumerate(payload.receiver_cache)},
                            cache_file)
                    meta = {
                        "task_id": acb.task_id, "cmd_ids": cmd_ids,
                        "hidden_context_tokens": payload.hidden_context_tokens,
                        "max_new_tokens": 640, "receiver_path": receiver_path,
                        "dtype": config["dtype"],
                        "attention_backend": config["attention_backend"],
                        "fusion_mode": "sequential",
                    }
                    if layout is not None:
                        meta["cache_layout"] = layout
                    meta_file.write_text(json.dumps(meta, ensure_ascii=False),
                                         encoding="utf-8")
                    page_fault, swap_in_seconds = self._pool_touch(acb, cache_file)
                    publish_seconds = perf_counter() - payload_started
                    iso_started = perf_counter()
                    worker_argv = [sys.executable, "-m", "agent_runtime.worker_entry",
                                   "--meta", str(meta_file), "--cache", str(cache_file),
                                   "--out", str(out_file)]
                    if payload_mmap:
                        worker_argv.append("--mmap")
                    run = run_isolated(
                        worker_argv,
                        IsolationLimits(memory_mib=worker_memory_mib,
                                        cpu_quota_pct=100, pids_max=128,
                                        timeout_s=worker_timeout_s),
                        env=child_env)
                    isolation_overhead = perf_counter() - iso_started
                    if run.rc != 0 or not out_file.exists():
                        return {"artifact": None, "gate": None,
                                "error": (f"isolated worker failed: rc={run.rc} "
                                          f"oom_killed={run.oom_killed} "
                                          f"timed_out={run.timed_out} "
                                          f"isolation={run.isolation} "
                                          f"stderr={run.error}"),
                                "wait_states": {"WAIT_MODEL": isolation_overhead,
                                                "WAIT_FUSION": publish_seconds}}
                    arm = json.loads(out_file.read_text(encoding="utf-8"))
                except Exception as exc:
                    return {"artifact": None, "gate": None,
                            "error": f"{type(exc).__name__}: {exc}",
                            "wait_states": {"WAIT_MODEL": 0.0}}
                arm.update({
                    "arm": f"acb_{acb.task_id}",
                    "command": acb.command,
                    "input_ids": payload_info["cmd_ids"],
                    "worker_visible_text": "public state + short command read natively; "
                       "Planner transcript text fully excluded (hidden "
                       "analysis KV only)",
                    "worker_input_construction": "state_native",
                    "hidden_context_tokens": payload.hidden_context_tokens,
                    "hidden_selection_source_indices": payload_info["selection"],
                    "hidden_text_ids": payload_info["hidden_text_ids"],
                    "hidden_reference_is_receiver_encoding_of_same_text": True,
                    "source_cache_sha256": payload_info["source_cache_sha256"],
                    "receiver_cache_sha256": payload_info["receiver_cache_sha256"],
                    "receiver_cache_bytes": payload_info["receiver_cache_bytes"],
                    "reference_prefill_seconds": payload_info["reference_prefill_seconds"],
                    "fusion_seconds": payload_info["fusion_seconds"],
                    "payload_publish_seconds": publish_seconds,
                    "isolation": {
                        "backend": run.isolation, "rc": run.rc,
                        "oom_killed": run.oom_killed, "timed_out": run.timed_out,
                        "wall_s": run.wall_s, "unit": run.unit,
                        "memory_peak_mib": run.memory_peak_mib,
                        "memory_enforcement": (run.extra or {}).get(
                            "memory_enforcement"),
                    },
                    "isolation_overhead_seconds": isolation_overhead,
                    "semantic_status": "unjudged_pending_offline_audit",
                    "payload_pool": {
                        "page_fault": page_fault,
                        "swap_in_seconds": swap_in_seconds,
                        "resident_after": list(self.payload_pool),
                        "capacity": payload_pool_capacity,
                        "mmap_publish": payload_mmap,
                    },
                })
                gate = _os_semantic_gate(case, arm, acb.command)
                arm["application_gate"] = gate
                arm["semantic_status"] = (
                    "accepted_by_public_os_gate" if gate["accepted"]
                    else "rejected_by_public_os_gate")
                (arms_dir / f"{acb.task_id}.json").write_text(
                    json.dumps(arm, ensure_ascii=False, indent=1), encoding="utf-8")
                wait_model = float(arm.get("service_seconds") or 0.0) \
                    + isolation_overhead
                wait_fusion = float(payload_info["fusion_seconds"] or 0.0) \
                    + float(payload_info["reference_prefill_seconds"] or 0.0) \
                    + publish_seconds
                return {"artifact": arm.get("output"), "gate": gate, "error": None,
                        "record": arm,
                        "wait_states": {"WAIT_MODEL": wait_model,
                                        "WAIT_FUSION": wait_fusion}}

        events = []
        controller = None
        backend = IsolatedBackend()
        if sim_quota is not None:
            from .resource_control import (ControlledBackend, ResourceController,
                                           SimulatedCgroupFs, SliceQuota, TaskQuota)
            cgroupfs = SimulatedCgroupFs(output / "cgroups")
            controller = ResourceController(
                cgroupfs,
                task_quota=TaskQuota(
                    tokens=sim_quota.get("tokens"), wall_s=sim_quota.get("wall_s"),
                    memory_mib=sim_quota.get("memory_mib")),
                slice_quota=SliceQuota(pids_max=sim_quota.get("slice_pids_max")),
                emit=events.append)
            backend = ControlledBackend(backend, controller)
        if sched_policy == "fair":
            from .os_sched import FairScheduler
            scheduler = FairScheduler(backend, repair_policy=repair_policy,
                                      emit=events.append)
        else:
            scheduler = ACBScheduler(backend, repair_policy=repair_policy,
                                     emit=events.append)
        result["sched_policy"] = sched_policy
        result["simulated_cgroup"] = sim_quota is not None

        def _map_deps(raw_deps, index):
            mapped = []
            for d in raw_deps or []:
                d = str(d).strip().lower()
                if d.startswith("t") and d[1:].isdigit() \
                        and 1 <= int(d[1:]) <= len(tasks) and int(d[1:]) != index + 1:
                    mapped.append(f"t{int(d[1:])}")
            return tuple(mapped)

        for i, task in enumerate(tasks):
            acb = AgentControlBlock(
                task_id=f"t{i + 1}",
                command=_worker_command(task),
                deliverable=str(task.get("deliverable", "")),
                depends_on=_map_deps(task.get("depends_on"), i),
                priority=float(len(tasks) - i),
                gpu_bytes_est=est_bytes,
                est_tokens=int(analysis.metrics.get("generated_tokens") or 0),
                max_attempts=2)
            scheduler.submit(acb)

        report = scheduler.run()
        result["acb_report"] = report
        result["scheduler_events"] = events
        result["payload_pool_events"] = pool_events
        result["payload_pool_capacity"] = payload_pool_capacity
        result["payload_mmap"] = payload_mmap
        result["arms"] = {"c2c_hidden_analysis": {
            "calls": list(report["tasks"].values()),
            "note": "each ACB task executed in an isolated OS child process "
                    "(worker_entry) consuming a published fused KV payload; "
                    "per-task cache/meta/records under tasks/<id>.*"}}
        result.update(status="completed",
                      wall_seconds=perf_counter() - started)
        if controller is not None:
            result["resource_control"] = controller.report()
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      wall_seconds=perf_counter() - started)
        raise
    finally:
        (output / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
