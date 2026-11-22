"""Frozen block-order comparison using the existing OS runtime and model host."""
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

from .project import ROOT
from .records import fingerprint

ORDERS = ("paper_first", "goal_first")
REPEATS = 3


def run_order_study(output_dir, *, interface_v2=False):
    from .models import ModelHost, METHODS, PlanningContractError
    from .runtime import execute
    data_dir = ROOT / "data/qasper-order-dev-16"
    manifest = json.loads((data_dir / "manifest.json").read_text())
    public = data_dir / "public.json"
    if hashlib.sha256(public.read_bytes()).hexdigest() != manifest["public_sha256"]:
        raise ValueError("Frozen public data changed")
    protocol_path = ROOT / ("configs/interface-study.json" if interface_v2 else "configs/goal-order-study.json")
    protocol = json.loads(protocol_path.read_text())
    orders = ("paper_first",) if interface_v2 else ORDERS
    output_contract = "evidence_ids_v2" if interface_v2 else "object_v1"
    if protocol["orders"] != list(orders) or (interface_v2 and protocol["output_contract"] != output_contract):
        raise ValueError("Frozen protocol order or interface changed")
    previous_path = ROOT / "artifacts/interface-20261005/previous-identities.json"
    previous = json.loads(previous_path.read_text()) if interface_v2 else None
    if protocol["public_sha256"] != manifest["public_sha256"]:
        raise ValueError("Public inputs differ from preregistration")
    papers = json.loads(public.read_text(encoding="utf-8"))
    if len(papers) != 16 or any(len(p["questions"]) != 2 for p in papers):
        raise ValueError("This comparison requires the frozen 16-paper/32-question denominator")
    config = json.loads((ROOT / "configs/cloud.json").read_text())
    lock = json.loads((ROOT / "configs/cloud.models.lock.json").read_text())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    result = {"kind": "interface_study" if interface_v2 else "order_study",
              "scope": ("16 frozen development papers; exact evidence-array interface alias; original paper-first application"
                        if interface_v2 else "16 new development papers; block order intervention; three text/C2C multi-agent arms"),
              "data_directory": "data/qasper-order-dev-16", "manifest": manifest,
              "orders": list(orders), "method_names": list(METHODS), "repeats": REPEATS,
              "output_contract": output_contract, "protocol": protocol,
              "expected_paper_ids": [p["id"] for p in papers], "preparations": [],
              "workflows": [{"paper_id": p["id"], "order": order, "method": method,
                    "repeat": repeat, "status": "not_run", "expected_questions": [q["id"] for q in p["questions"]]}
                    for p in papers for order in orders for repeat in range(REPEATS) for method in METHODS],
              "status": "running", "content_takeovers": 0,
              "timing_policy": {
                  "raw_execution": "measured outer wall of each fresh two-question OS workflow, including failed and blocked nodes",
                  "setup": "one actual Planner and one actual handoff per paper/order; stage samples are not three independent setup measurements",
                  "new_background_estimate": "common planning + method setup + execution wall with measured log time subtracted; attribution estimate, not a separately measured end-to-end run",
                  "replay": "three fresh OS workflows branch from each method's cached background; artifacts never reused",
                  "method_order": "Latin rotation by paper index plus repeat; protocol orders alternate by paper index when both present",
                  "warmup": "no discarded warmup; first-use repeat and later repeats retained; models already exercised by shared preparation",
                  "overlap": "request wait and process lifetime overlap service; do not add their sums to outer wall",
                  "memory": "all models/arms co-resident; absolute peak is development-process memory, not single-arm deployment",
                  "loading": "measured once for the whole batch, not once per task"}}
    started, logging_seconds = perf_counter(), 0.0
    identity, traces = {}, {}
    log = (output_dir / "events.jsonl").open("x", encoding="utf-8")

    def emit(kind, **fields):
        nonlocal logging_seconds
        before = perf_counter()
        if kind == "model_result":
            generation = fields["generation"]
            traces[fields["task_id"]] = {"input_sha256": fingerprint(generation["input_ids"]),
                "command_sha256": fingerprint(generation["command"]),
                "generated_sha256": fingerprint(generation["generated_token_ids"]),
                "output": generation["output"], "stop_reason": generation["stop_reason"]}
        log.write(json.dumps({"type": kind, **identity, **fields}, ensure_ascii=False, allow_nan=False) + "\n")
        log.flush()
        logging_seconds += perf_counter()-before

    def save_progress():
        # Replaces only this run's progress file, never records from another run.
        temporary = output_dir / "progress.json.tmp"
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output_dir / "progress.json")

    sources = [*sorted((ROOT / "src/agent_runtime").glob("*.py")), ROOT / "scripts/run.py",
               ROOT / "scripts/prepare_data.py", protocol_path]
    if interface_v2:
        sources.append(previous_path)
    emit("metadata", config=config, models=lock, manifest=manifest, pid=os.getpid(),
         protocol=protocol,
         source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
         expected_workflows=len(result["workflows"]), worker_limit=2, reference_answers_accessible=False,
         output_contract=output_contract,
         content_recovery=False, worker_constraints=False, physical_zero_copy=False,
         cgroup_enabled=False, separate_gpu_owner_process=False, timing_policy=result["timing_policy"])
    save_progress()
    try:
        loaded_at = perf_counter()
        host = ModelHost(config, lock)
        result["model_loading_seconds"] = perf_counter()-loaded_at
        import torch, transformers
        emit("environment", torch=str(torch.__version__), transformers=transformers.__version__,
             gpu=torch.cuda.get_device_name(), model_loading_seconds=result["model_loading_seconds"])
        for paper_index, paper in enumerate(papers):
            for order in (orders if paper_index % 2 == 0 else tuple(reversed(orders))):
                identity = {"paper_id": paper["id"], "order": order}
                prep = {**identity, "status": "running"}
                result["preparations"].append(prep)
                before, log_before = perf_counter(), logging_seconds
                try:
                    plan = host.prepare(paper, emit, order=order)
                except Exception as exc:
                    prep.update(status="failed", observed_wall_seconds=perf_counter()-before,
                                logging_seconds=logging_seconds-log_before,
                                error={"type": type(exc).__name__, "message": str(exc)})
                    emit("preparation_failed", **prep)
                    for row in result["workflows"]:
                        if row["paper_id"] == paper["id"] and row["order"] == order:
                            row.update(status="preparation_failed", error=prep["error"],
                                       paid_preparation_wall_seconds=prep["observed_wall_seconds"])
                    save_progress()
                    if isinstance(exc, PlanningContractError):
                        continue
                    raise  # Identity/implementation failures cannot silently become quality results.
                prep.update(status="completed", plan=plan, handle=host.handle, hashes=dict(host.hashes),
                            observed_wall_seconds=perf_counter()-before, logging_seconds=logging_seconds-log_before,
                            planning_seconds=host.planning_seconds, method_setup_seconds=dict(host.costs),
                            metrics=host.preparation_metrics)
                if interface_v2:
                    reference = previous[paper["id"]]
                    if fingerprint(host.prefix) != reference["prefix_sha256"] or host.hashes != reference["cache_hashes"]:
                        raise RuntimeError("Interface verification changed actual planning or published cache identity")
                    prep["previous_actual_cache_identity_unchanged"] = True
                emit("preparation_measured", preparation=prep)
                arm_traces, repeated_traces = {}, {}
                for repeat in range(REPEATS):
                    shift = (paper_index + repeat) % len(METHODS)
                    sequence = METHODS[shift:] + METHODS[:shift]
                    for method in sequence:
                        identity = {"paper_id": paper["id"], "order": order, "method": method, "repeat": repeat}
                        row = next(r for r in result["workflows"] if all(r[k] == v for k, v in identity.items()))
                        if method not in host.payloads:
                            row.update(status="handoff_failed", reason="Actual handoff incomplete; no replacement or truncation",
                                       paid_setup_seconds=host.planning_seconds+host.costs[method])
                            emit("workflow_unavailable", **row)
                            continue
                        traces = {}
                        row["status"] = "running"
                        execution_started = perf_counter()
                        try:
                            execution = execute(paper, plan, host, method, emit, output_contract=output_contract)
                        except Exception as exc:
                            row.update(status="execution_failed", observed_execution_wall_seconds=perf_counter()-execution_started,
                                       error={"type": type(exc).__name__, "message": str(exc)}, trace=dict(traces))
                            emit("workflow_failed", workflow=row)
                            raise
                        row["observed_execution_wall_seconds"] = perf_counter()-execution_started
                        row.update(status="completed", execution=execution, trace=dict(traces),
                            new_background_attribution_seconds=host.planning_seconds+host.costs[method]
                                + row["observed_execution_wall_seconds"]-execution["metrics"]["logging_seconds"])
                        if interface_v2:
                            replay = []
                            for task_id, trace in traces.items():
                                if not task_id.startswith("evidence:"):
                                    continue
                                old = previous[paper["id"]]["retrievals"][method+"/"+task_id]
                                if any(trace[k] != old[k] for k in ("input_sha256", "command_sha256")):
                                    raise RuntimeError("Interface verification changed the retrieval command or input")
                                replay.append({"task_id": task_id, "same_input": True,
                                               "same_generated_tokens": trace["generated_sha256"] == old["generated_sha256"]})
                            row["previous_retrieval_replay"] = replay
                        if execution["metrics"]["non_model_wall_seconds_including_tools"] < -1e-6:
                            raise RuntimeError("Timing decomposition overlaps or exceeds measured execution wall")
                        if execution["metrics"]["model_errors"]:
                            raise RuntimeError("Model service error; stop and inspect instead of hiding an instrumentation failure")
                        if method in repeated_traces and traces != repeated_traces[method]:
                            row["repeat_generation_changed"] = True
                            emit("repeat_generation_changed", original_repeat=0)
                        repeated_traces.setdefault(method, dict(traces))
                        arm_traces[method] = dict(traces)
                        for task_id, trace in traces.items():
                            if not task_id.startswith("evidence:"):
                                continue
                            for other, other_trace in arm_traces.items():
                                if task_id in other_trace:
                                    if trace["command_sha256"] != other_trace[task_id]["command_sha256"]:
                                        raise RuntimeError("Methods did not receive identical retrieval commands")
                                    if {method, other} == {"text_full", "c2c_shared"} and trace["input_sha256"] != other_trace[task_id]["input_sha256"]:
                                        raise RuntimeError("Native and fused arms did not receive identical real history")
                        emit("workflow_measured", workflow=row)
                        print(paper["id"], order, method, repeat, "delivered",
                              sum(not r.get("failed") for r in execution["report"]),
                              "wall", round(execution["wall_seconds"], 3), flush=True)
                    before = perf_counter()
                    audit = host.audit()
                    seconds = perf_counter()-before
                    result.setdefault("context_audits", []).append({"paper_id": paper["id"], "order": order,
                        "repeat": repeat, "seconds": seconds, **audit})
                    emit("context_audit", seconds=seconds, **audit)
                    save_progress()
        result["status"] = "completed"
    except Exception as exc:
        result.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        emit("run_error", **result["error"])
        raise
    finally:
        result["observed_batch_wall_seconds"] = perf_counter()-started
        result["event_logging_seconds"] = logging_seconds
        result["wall_scope"] = "one actual load, all actual preparations, three executions per arm, audits and progress writes; final results serialization excluded"
        with (output_dir / "results.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
        emit("run_finished", status=result["status"])
        log.close()
