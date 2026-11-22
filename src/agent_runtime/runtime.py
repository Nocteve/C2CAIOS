"""Two real OS workers, dependency artifacts and bounded dynamic repair."""
import json
import multiprocessing as mp
import os
from time import perf_counter
from multiprocessing.connection import wait
from .records import fingerprint
from .workload import search_command, answer_command


def validate_output(generated, job):
    """Validate freely decoded output; a codec never repairs semantic content."""
    if generated.get("error"):
        raise ValueError(generated["error"])
    if generated["stop_reason"] != "eos":
        raise ValueError("Model reached its output limit")
    output = generated["output"].strip()
    if job["kind"] == "locate":
        contract = job.get("output_contract", "object_v1")
        if contract not in {"object_v1", "evidence_ids_v2"}:
            raise ValueError("Unknown evidence output contract")
        result = json.loads(output)
        if isinstance(result, dict) and set(result) == {"paragraphs"}:
            ids, form = result["paragraphs"], "object"
        elif contract == "evidence_ids_v2" and isinstance(result, list):
            ids, form = result, "array"
        else:
            raise ValueError("Expected an evidence-ID object or an enabled exact array alias")
        if not isinstance(ids, list) or len(ids) > 6:
            raise ValueError("Expected at most six original paragraph IDs")
        if any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids) or set(ids) - set(job["valid_ids"]):
            raise ValueError("Unknown or duplicate evidence IDs")
        value = {"paragraph_ids": ids}
        if contract == "evidence_ids_v2":
            value["validation"] = {"contract": contract, "accepted_form": form,
                                   "original_object_contract_pass": form == "object",
                                   "semantic_verified": False, "content_repaired": False}
        return value
    if not output:
        raise ValueError("Empty answer")
    return {"answer": output}


def worker(connection, job):
    # Limits apply only to this process, never the user's host configuration.
    if os.name == "posix":
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
        resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    try:
        connection.send({"type": "started", "pid": os.getpid()})
        connection.send({"type": "model_request", "command": job["command"],
                         "requested_at": perf_counter(),
                         "context": "paper" if job["kind"] == "locate" else "evidence_only",
                         "budget": 96 if job["kind"] == "locate" else 128})
        generated = connection.recv()
        result = validate_output(generated, job)
        connection.send({"type": "result", "value": result, "pid": os.getpid()})
    except Exception as exc:
        connection.send({"type": "failed", "error": f"{type(exc).__name__}: {exc}", "pid": os.getpid()})
    finally:
        connection.close()


def execute(paper, plan, host, method, emit, inject_exit=False, *, output_contract="evidence_ids_v2"):
    """Scheduler owns paper/tools; answer processes receive only evidence artifacts."""
    if output_contract not in {"object_v1", "evidence_ids_v2"}:
        raise ValueError("Unknown evidence output contract")
    outer_emit, logging_seconds = emit, 0.0
    def emit(kind, **fields):
        nonlocal logging_seconds
        before = perf_counter()
        outer_emit(kind, **fields)
        logging_seconds += perf_counter() - before
    metrics = {"model_call_seconds": 0.0, "model_service_seconds": 0.0,
               "task_prefill_seconds_including_clone": 0.0, "cache_clone_seconds": 0.0,
               "decode_seconds": 0.0, "tool_seconds": 0.0, "generated_tokens": 0,
               "model_calls": 0, "model_errors": 0, "model_first_token_seconds": [],
               "request_wait_seconds": [], "peak_allocated_mib": 0.0,
               "peak_incremental_request_mib": 0.0, "requests": []}
    ctx = mp.get_context("spawn")
    questions = {q["id"]: q["text"] for q in paper["questions"]}
    paragraphs = {p["id"]: p for p in paper["paragraphs"]}
    tasks, artifacts, live = {}, {}, {}
    started = perf_counter()
    for assignment in plan:
        q = assignment["question_id"]
        eid, aid = "evidence:" + q, "answer:" + q
        tasks[eid] = {"id": eid, "kind": "locate", "question_id": q, "deps": [],
            "state": "READY", "command": search_command(questions[q], assignment["focus"])
                + "\nValid original paragraph IDs: " + ", ".join(paragraphs),
            "valid_ids": list(paragraphs), "repairs": 0, "output_contract": output_contract}
        tasks[aid] = {"id": aid, "kind": "answer", "question_id": q, "deps": [eid],
            "state": "WAIT_DEPS", "repairs": 0}
    emit("dag_created", tasks=list(tasks.values()), workers=2)

    def fail(task, reason, *, infrastructure=False):
        task["state"] = "FAILED"
        emit("task_failed", task_id=task["id"], reason=reason)
        if not inject_exit or not infrastructure or task["repairs"] >= 1:
            return
        # A bounded process retry cannot create or alter semantic instructions.
        rid = task["id"] + ":repair"
        replacement = {**task, "id": rid, "state": "READY", "repairs": 1}
        tasks[rid] = replacement
        for consumer in tasks.values():
            if consumer["id"] != rid:
                consumer["deps"] = [rid if d == task["id"] else d for d in consumer["deps"]]
        emit("repair_node_added", failed_task=task["id"], new_task=replacement,
             action="retry_identical_task", content_takeover=False, planner_called=False,
             original_command_sha256=fingerprint(task["command"]),
             replacement_command_sha256=fingerprint(replacement["command"]))

    def finish_process(connection):
        process, task, began, _ = live.pop(connection)
        process.join(timeout=2)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join(timeout=2)
        connection.close()
        emit("process_reaped", task_id=task["id"], pid=process.pid,
             exitcode=process.exitcode, alive=process.is_alive())
        task.update(pid=process.pid, exitcode=process.exitcode,
                    process_lifetime_seconds=perf_counter()-began)
        if process.is_alive():
            raise RuntimeError(f"Could not reap worker {process.pid}")
        process.close()

    injected = False
    try:
        while True:
            for task in list(tasks.values()):
                if task["state"] == "WAIT_DEPS" and all(d in artifacts for d in task["deps"]):
                    evidence = artifacts[task["deps"][0]]["value"]["passages"]
                    task["command"] = answer_command(questions[task["question_id"]], evidence)
                    task["state"] = "READY"
                if task["state"] != "READY" or len(live) >= 2:
                    continue
                parent, child = ctx.Pipe()
                process = ctx.Process(target=worker, args=(child, task), name=task["id"])
                began = perf_counter()
                process.start()
                child.close()
                task["state"] = "RUNNING"
                live[parent] = (process, task, began, False)
                emit("task_started", task_id=task["id"], pid=process.pid,
                     input_artifacts=[artifacts[d]["sha256"] for d in task["deps"]],
                     context_access="paper_handle" if task["kind"] == "locate" else "evidence_artifact_only")
            if not live:
                break
            for connection in wait(list(live), timeout=0.1):
                process, task, began, _ = live[connection]
                try:
                    message = connection.recv()
                except (EOFError, OSError) as exc:
                    reason = f"Worker IPC closed without result ({type(exc).__name__}, exit={process.exitcode})"
                    finish_process(connection)
                    fail(task, reason, infrastructure=True)
                    continue
                if message["type"] == "model_request":
                    task["state"] = "WAIT_LLM"
                    emit("model_wait", task_id=task["id"], handle=host.handle if task["kind"] == "locate" else None)
                    if inject_exit and not injected:
                        process.terminate()
                        process.join(timeout=2)
                        injected = True
                        emit("injected_worker_exit", task_id=task["id"], pid=process.pid,
                             exitcode=process.exitcode, phase="model request in flight")
                    call_started = perf_counter()
                    metrics["request_wait_seconds"].append(call_started-message["requested_at"])
                    metrics["model_calls"] += 1
                    try:
                        generation = host.infer(method, **{k: message[k] for k in ("command", "budget", "context")})
                    except Exception as exc:
                        metrics["model_errors"] += 1
                        generation = {"error": str(exc)}
                    finally:
                        metrics["model_call_seconds"] += perf_counter()-call_started
                    if not generation.get("error"):
                        metrics["model_service_seconds"] += generation["service_seconds"]
                        metrics["task_prefill_seconds_including_clone"] += generation["receiver_task_prefill_seconds"]
                        metrics["cache_clone_seconds"] += generation["task_clone_seconds"]
                        metrics["decode_seconds"] += generation["decode_seconds"]
                        metrics["generated_tokens"] += generation["output_tokens"]
                        metrics["model_first_token_seconds"].append(generation["first_token_seconds"])
                        metrics["peak_allocated_mib"] = max(metrics["peak_allocated_mib"], generation["peak_allocated_mib"])
                        metrics["peak_incremental_request_mib"] = max(metrics["peak_incremental_request_mib"], generation["peak_incremental_mib"])
                        metrics["requests"].append({"task_id": task["id"], "kind": task["kind"],
                            **{k: generation[k] for k in ("service_seconds", "first_token_seconds",
                                "receiver_task_prefill_seconds", "task_clone_seconds", "decode_seconds",
                                "output_tokens", "prompt_tokens", "cached_prefix_tokens", "stop_reason")}})
                        emit("model_result", task_id=task["id"], generation=generation)
                    else:
                        emit("model_error", task_id=task["id"], error=generation["error"])
                    try:
                        connection.send(generation)
                    except (BrokenPipeError, EOFError, OSError) as exc:
                        finish_process(connection)
                        fail(task, f"Worker communication closed during model request: {type(exc).__name__}", infrastructure=True)
                        continue
                    task["state"] = "RUNNING"
                elif message["type"] == "result":
                    tool_started = perf_counter()
                    value = message["value"]
                    if task["kind"] == "locate":
                        # Exact public IDs only. No gold, fuzzy lookup, or repair.
                        value["passages"] = [paragraphs[i] for i in value["paragraph_ids"]]
                    artifact = {"producer": task["id"], "value": value,
                                "sha256": fingerprint({"producer": task["id"], "value": value})}
                    artifacts[task["id"]] = artifact
                    metrics["tool_seconds"] += perf_counter()-tool_started
                    task["state"] = "DONE"
                    emit("artifact_created", artifact=artifact, pid=message["pid"], semantic_verified=False)
                    finish_process(connection)
                elif message["type"] == "failed":
                    fail(task, message["error"])
                    finish_process(connection)
            for connection, (process, task, began, _) in list(live.items()):
                if perf_counter() - began > 180:
                    process.terminate()
                    process.join(timeout=2)
                    fail(task, "Worker deadline exceeded", infrastructure=True)
                    finish_process(connection)
        for task in tasks.values():
            if task["state"] == "WAIT_DEPS":
                task["state"] = "BLOCKED"
                emit("task_blocked", task_id=task["id"], dependencies=task["deps"])
        report = []
        for question_id, question in questions.items():
            answers = [t for t in tasks.values() if t["kind"] == "answer" and t["question_id"] == question_id and t["state"] == "DONE"]
            if answers:
                answer_task = answers[-1]
                evidence = artifacts[answer_task["deps"][0]]
                answer = artifacts[answer_task["id"]]
                report.append({"question_id": question_id, "question": question,
                    "answer": answer["value"]["answer"], "evidence": evidence["value"],
                    "evidence_sha256": evidence["sha256"], "answer_sha256": answer["sha256"]})
            else:
                report.append({"question_id": question_id, "question": question, "answer": "", "failed": True})
            # Evidence quality is independent of whether the answer was delivered.
            evidence_tasks = [t for t in tasks.values() if t["kind"] == "locate"
                              and t["question_id"] == question_id and t["state"] == "DONE"]
            item = report[-1]
            item["evidence_delivered"] = bool(evidence_tasks)
            if evidence_tasks:
                evidence = artifacts[evidence_tasks[-1]["id"]]
                item["evidence"] = evidence["value"]
                item["evidence_sha256"] = evidence["sha256"]
            else:
                item["evidence"] = {"paragraph_ids": [], "passages": []}
        wall = perf_counter()-started
        metrics.update(logging_seconds=logging_seconds,
                       wall_seconds_excluding_log_estimate=wall-logging_seconds,
                       non_model_wall_seconds_including_tools=wall-logging_seconds-metrics["model_call_seconds"],
                       interval_sums_are_not_additive_to_wall=True,
                       memory_scope="all comparison models and caches co-resident; not single-arm deployment memory")
        return {"report": report, "tasks": list(tasks.values()), "wall_seconds": wall, "metrics": metrics,
                "output_contract": output_contract,
                "recovery_policy": "one identical infrastructure retry" if inject_exit else "none",
                "content_takeovers": 0,
                "repair_nodes": sum(t["repairs"] for t in tasks.values()), "injected_exit": injected}
    finally:
        for connection in list(live):
            finish_process(connection)
