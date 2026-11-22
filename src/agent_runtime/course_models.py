"""GPU owner for the course runtime. One serialized GPU queue, real AR caches.

Task processes supervise requests; model computation and its costs belong here.
No claim of concurrent GPU generation, zero-copy or one-fusion-many-workers.
"""
import json
import os
import traceback
from pathlib import Path
from time import perf_counter


def action_schema(case, max_tasks):
    from .c2c_core import _core_schema
    from .delegation_contract import object_schema
    text = {"type": "string", "minLength": 1, "maxLength": 700}
    ids = [j["id"] for j in case["public_background"]["tasks"]]
    job = {"type": "string", "enum": ids}
    report = {"type": "object", "properties": {
        "start": {"type": "array", "items": job, "maxItems": len(ids)},
        "wait": {"type": "object", "properties": {j: {"type": "array", "items": job, "maxItems": len(ids)} for j in ids}, "additionalProperties": False},
        "monitor": {"type": "object", "properties": {j: {"type": "integer", "minimum": 0, "maximum": 1000000} for j in ids}, "additionalProperties": False},
        "free_after": object_schema({k: {"type": "integer", "minimum": 0, "maximum": 1000000000} for k in ("cpu", "gpu", "memory_mb")}),
    }, "additionalProperties": False}
    schema = _core_schema(max_tasks)
    report_action = object_schema({"action": {"const": "report"}, "report": report, "reason": text})
    report_action["properties"]["evidence_task_ids"] = {"type":"array","items":{"type":"string"},"maxItems":8}
    schema["anyOf"] += [object_schema({"action": {"const": "wait"}}),
        object_schema({"action": {"const": "accept"}, "task_id": text}),
        object_schema({"action": {"const": "cancel"}, "task_id": text, "reason": text}),
        report_action]
    return schema


def _record(t, digest):
    return dict(output=t.output, prompt_ids=t.prompt_ids, generated_ids=t.generated_ids,
                prefix_ids=t.prefix_ids, metrics=t.metrics, source_cache_sha256=digest(t.source_cache))


def serve(requests, responses, config, lock, output):
    started = perf_counter()
    try:
        from .loading import load_models
        from .cache_ops import cache_digest, prefill
        from .c2c_core import _messages, _run_hidden_arm, _run_arm, _worker_command
        from .planning_policy import analysis_messages, ACTION_MESSAGE, ARITHMETIC_SCAFFOLD_INSTRUCTION, FEEDBACK_REVIEW_INSTRUCTION
        from .transcript import capture_plan
        from .planner_tools import WORKER_ROLE_SYSTEM
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data, build_transformers_prefix_allowed_tokens_fn
        combined, tokenizers = load_models(config, lock)
        combined.multi_source_fusion_mode = "sequential"
        sender, receiver = combined.model_list[1], combined.model_list[0]
        publications = {}
        responses.put({"kind": "ready", "pid": os.getpid(), "load_seconds": perf_counter()-started})
        while True:
            req = requests.get()
            if req["kind"] == "stop":
                responses.put({"kind":"stopped","pid":os.getpid()})
                break
            if req["kind"] == "release":
                publications.pop(req["publication"], None)
                continue
            call_started = perf_counter()
            responses.put({"kind":"started","id":req["id"],"queue_seconds":call_started-req["submitted_monotonic"]})
            record = {"request": req, "service_pid": os.getpid(), "queue_seconds": call_started-req["submitted_monotonic"]}
            try:
                if req["kind"] == "planner":
                    case, context = req["case"], req["context"]
                    messages = _messages(case, max_tasks=req["max_tasks"], remaining_decisions=context["remaining_decisions"])
                    content = json.loads(messages[-1]["content"])
                    content.update(goal=case["goal"], course_runtime=context)
                    content["instructions"] += (
                        " You own a SCHEDULING PLAN artifact, not execution of the snapshot business programs. "
                        "Available additional actions: report(report,reason) writes your own proposed plan, accept(task_id) "
                        "adopts a validated Worker proposal, cancel(task_id,reason), wait. A report replaces the entire current plan; "
                        "partial Worker proposals merge only by your explicit accept and cannot overwrite conflicting fields. "
                        "Complete the four plan fields described in course_runtime.plan_interface before finish completed. "
                        "Only your final plan uses that interface. Workers return their assigned local reports freely, not necessarily plan objects. "
                        "You may optionally name evidence_task_ids in your report to explicitly cite actual REPORTED Worker evidence. "
                        "REPORTED means delivered, not semantically verified. Agent dependencies wait for actual report delivery; public job prerequisites require terminal-success. "
                        "Dependencies of Agent assignments use agent:<issued ID>; public job prerequisites use job:<ID>. "
                        "A diagnosis OF a blocked public job need not wait for that job to finish. "
                        "The earlier runnable-only restriction applies to public business jobs; Agent assignments may depend on pending Agent tasks and wait in the Agent DAG. "
                        "Use wait for in-flight results; cancel or take over failed work when useful. "
                        "The host does not choose a decomposition. Self-execution and zero delegation are valid.")
                    messages[-1]["content"] = json.dumps(content, ensure_ascii=False)
                    extra = ARITHMETIC_SCAFFOLD_INSTRUCTION
                    if context["worker_observation"]:
                        extra += FEEDBACK_REVIEW_INSTRUCTION + (
                            " In this application goal dependencies mean the actual Agent plan obligations, NOT completion of immutable public business jobs. "
                            "For process failures with no report, review the actual failure evidence instead of quoting nonexistent report facts. "
                            "A rejected proposal may be replaced by your own validated report; it does not permanently forbid finish.")
                    note = capture_plan(sender, tokenizers["sender"], tokenizers["receiver"],
                                        analysis_messages(messages, extra_instruction=extra), 1024 if req["max_tasks"] > 1 else 512,
                                        config["max_input_tokens"], producer_role="Planner course analysis")
                    record["analysis"] = _record(note, cache_digest)
                    if not note.metrics["complete_assistant_turn"] or not note.output.strip():
                        raise ValueError("incomplete_planner_analysis")
                    allowed = build_transformers_prefix_allowed_tokens_fn(
                        build_token_enforcer_tokenizer_data(tokenizers["sender"]), JsonSchemaParser(action_schema(case, req["max_tasks"])))
                    # New report/accept actions require room beyond the frozen two-task 256-token diagnostic.
                    t = capture_plan(sender, tokenizers["sender"], tokenizers["receiver"], ACTION_MESSAGE,
                                     768, config["max_input_tokens"], producer_role="Planner course action",
                                     prior_prefix_ids=note.prefix_ids, prior_cache=note.source_cache, allowed_tokens=allowed)
                    t.planning_input_ids, t.planning_input_text = note.prompt_ids, note.prompt_text
                    t.analysis_token_range = (len(note.prompt_ids), len(note.prompt_ids)+len(note.generated_ids)-1)
                    record["action"] = _record(t, cache_digest)
                    if not t.metrics["complete_assistant_turn"]:
                        raise ValueError("incomplete_planner_action")
                    action = json.loads(t.output)
                    if action["action"] == "delegate":
                        publications[req["id"]] = {"transcript": t, "receiver_cache": None}
                    response = {"action": action, "publication": req["id"]}
                else:
                    p = publications[req["publication"]]
                    t = p["transcript"]
                    header, body = _worker_command(req["task"]).split("WORKER CONTRACT JSON:\n",1)
                    contract = json.loads(body)
                    # Preserve the original native role instructions, replacing fixture metadata once.
                    contract.update(req["control"])
                    command = header + "WORKER CONTRACT JSON:\n" + json.dumps(contract, ensure_ascii=False)
                    if req["method"] == "c2c_hidden_analysis":
                        lo, hi = t.analysis_token_range
                        generated = _run_hidden_arm(combined, tokenizers, t, receiver, command, tuple(range(lo,hi)), "c2c_hidden_analysis")
                    elif req["method"] == "text_full":
                        prepare = perf_counter()
                        reused = p["receiver_cache"] is not None
                        if not reused:
                            p["receiver_cache"], _ = prefill(receiver, t.prefix_ids)
                        generated, _ = _run_arm(combined, tokenizers, t, t.prefix_ids, p["receiver_cache"], command, "text_full")
                        generated.update(receiver_prefix_reused=reused, full_call_including_required_prefill_seconds=perf_counter()-prepare)
                    else:
                        handoff_request = json.dumps({"invocation_role": "Planner", "current_operation": "prepare_worker_handoff",
                            "instructions": "Write a standalone useful handoff for the exact Worker task below. Include relevant public facts, constraints, dependency reasoning, evidence and working guidance. Useful supported calculations and answers are allowed. Distinguish missing information and unverified claims. Do not emit a scheduler action or merely repeat the command.",
                            "task": req["task"], "actual_dependency_artifacts": req["control"]}, ensure_ascii=False)
                        h = capture_plan(sender, tokenizers["sender"], tokenizers["receiver"], [{"role": "user", "content": handoff_request}],
                                         768, config["max_input_tokens"], producer_role="Planner practical T2T handoff",
                                         prior_prefix_ids=t.prefix_ids, prior_cache=t.source_cache)
                        record["handoff"] = _record(h, cache_digest)
                        if not h.metrics["complete_assistant_turn"] or not h.output.strip():
                            raise ValueError("incomplete_actual_T2T_handoff")
                        prefix = tokenizers["receiver"].apply_chat_template([
                            {"role": "system", "content": WORKER_ROLE_SYSTEM},
                            {"role": "user", "content": "Planner handoff:\n"+h.output}], tokenize=True, add_generation_prompt=False)
                        cache, seconds = prefill(receiver, prefix)
                        generated, _ = _run_arm(combined, tokenizers, t, prefix, cache, command, "text_brief", role_system=None, handoff_reencoded=True)
                        generated["receiver_prefill_seconds"] = seconds
                    record["worker"] = generated
                    response = {"generated": generated}
                record["wall_seconds"] = perf_counter()-call_started
                response.update(kind="result", id=req["id"], wall_seconds=record["wall_seconds"])
            except Exception:
                record["error"] = traceback.format_exc()
                record["wall_seconds"] = perf_counter()-call_started
                response = {"kind": "result", "id": req["id"], "error": record["error"], "wall_seconds": record["wall_seconds"]}
            audit_started = perf_counter()
            path = Path(output)/("model-"+req["id"]+".json")
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            response.update(raw_record=path.name, integrity_logging_seconds=perf_counter()-audit_started)
            responses.put(response)
    except Exception:
        responses.put({"kind": "fatal", "error": traceback.format_exc()})
