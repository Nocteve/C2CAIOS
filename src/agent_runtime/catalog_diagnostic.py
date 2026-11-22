"""Fixed reversal of the control ID catalog; no references or answer workers."""
import hashlib
import json
import os
from pathlib import Path
import re
from time import perf_counter

from .project import ROOT
from .records import fingerprint
from .workload import search_command


def catalog_observation(output, catalog):
    # Raw emitted IDs, including truncated outputs, are descriptive only.
    # They never become an accepted artifact or a repaired model response.
    ids = re.findall(r'"(p\d+)"', output)
    prefix = 0
    for left, right in zip(ids, catalog):
        if left != right:
            break
        prefix += 1
    ranks = {value: index for index, value in enumerate(catalog)}
    adjacent = [(ranks[a], ranks[b]) for a, b in zip(ids, ids[1:]) if a in ranks and b in ranks]
    return {"raw_quoted_ids": ids, "catalog_prefix_agreement": prefix,
            "adjacent_pairs": len(adjacent),
            "adjacent_steps_in_presented_order": sum(b == a+1 for a, b in adjacent),
            "descriptive_only_no_artifact_repair": True}


def run_catalog_diagnostic(output_dir):
    from .models import ModelHost, METHODS, PlanningContractError
    from .runtime import validate_output
    protocol_path = ROOT / "configs/catalog-order-diagnostic.json"
    protocol = json.loads(protocol_path.read_text())
    public_path = ROOT / "data/qasper-order-dev-16/public.json"
    if hashlib.sha256(public_path.read_bytes()).hexdigest() != protocol["public_sha256"]:
        raise ValueError("Frozen public inputs changed")
    papers = json.loads(public_path.read_text(encoding="utf-8"))[:2]
    if [p["id"] for p in papers] != protocol["paper_ids"] or any(len(p["questions"]) != 2 for p in papers):
        raise ValueError("Diagnostic fixed paper/question selection changed")
    if protocol["methods"] != list(METHODS) or protocol["expected_cases"] != 24 or protocol["evidence_budget"] != 96:
        raise ValueError("Diagnostic comparison contract changed")
    previous_path = ROOT / "artifacts/catalog-order-20261005/previous-identities.json"
    previous = json.loads(previous_path.read_text())
    config = json.loads((ROOT / "configs/cloud.json").read_text())
    lock = json.loads((ROOT / "configs/cloud.models.lock.json").read_text())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    result = {"kind": "catalog_order_diagnostic", "protocol": protocol, "status": "running",
        "cases": [{"paper_id": p["id"], "question_id": q["id"], "method": method,
                   "catalog_order": order, "status": "not_run", "contract_pass": False}
                  for p in papers for q in p["questions"] for method in METHODS
                  for order in protocol["catalog_orders"]],
        "preparations": [], "audits": [], "pid": os.getpid(), "content_takeovers": 0,
        "answer_successors": False, "os_workers": False, "reference_answers_read": False}
    log = (output_dir / "events.jsonl").open("x", encoding="utf-8")
    identity, logs = {}, 0.0
    def emit(kind, **fields):
        nonlocal logs
        start = perf_counter()
        log.write(json.dumps({"type": kind, **identity, **fields}, ensure_ascii=False, allow_nan=False)+"\n")
        log.flush()
        logs += perf_counter()-start
    sources = [*sorted((ROOT / "src/agent_runtime").glob("*.py")), ROOT / "scripts/run.py", protocol_path, previous_path]
    started = perf_counter()
    emit("metadata", protocol=protocol, config=config, model_pins=lock, pid=os.getpid(),
         source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    try:
        loading = perf_counter()
        host = ModelHost(config, lock)
        result["model_loading_seconds"] = perf_counter()-loading
        import torch, transformers
        emit("environment", torch=str(torch.__version__), transformers=transformers.__version__, gpu=torch.cuda.get_device_name())
        for pi, paper in enumerate(papers):
            identity = {"paper_id": paper["id"]}
            before, before_log = perf_counter(), logs
            try:
                plan = host.prepare(paper, emit, order="paper_first")
            except PlanningContractError as exc:
                result["preparations"].append({**identity, "status": "failed", "observed_wall_seconds": perf_counter()-before,
                    "logging_seconds": logs-before_log, "error": str(exc)})
                for row in result["cases"]:
                    if row["paper_id"] == paper["id"]:
                        row.update(status="preparation_failed", error=str(exc))
                continue
            prep = {**identity, "status": "completed", "observed_wall_seconds": perf_counter()-before,
                "logging_seconds": logs-before_log, "plan": plan, "handle": host.handle,
                "prefix_sha256": fingerprint(host.prefix), "cache_hashes": dict(host.hashes),
                "metrics": host.preparation_metrics, "method_setup_seconds": dict(host.costs)}
            result["preparations"].append(prep)
            reference = previous[paper["id"]]
            if prep["prefix_sha256"] != reference["prefix_sha256"] or host.hashes != reference["cache_hashes"]:
                raise RuntimeError("Recreated actual planning/cache identity differs from frozen baseline")
            emit("preparation_measured", preparation=prep, prior_actual_cache_identity_unchanged=True)
            focus = {task["question_id"]: task["focus"] for task in plan}
            valid_ids = [p["id"] for p in paper["paragraphs"]]
            for qi, question in enumerate(paper["questions"]):
                shift = (pi+qi) % len(METHODS)
                for mi, method in enumerate(METHODS[shift:]+METHODS[:shift]):
                    orders = protocol["catalog_orders"] if (pi+qi+mi) % 2 == 0 else list(reversed(protocol["catalog_orders"]))
                    for order in orders:
                        identity = {"paper_id": paper["id"], "question_id": question["id"], "method": method, "catalog_order": order}
                        row = next(r for r in result["cases"] if all(r[k] == value for k, value in identity.items()))
                        if method not in host.payloads:
                            row.update(status="handoff_failed", error="Actual handoff incomplete; no replacement")
                            continue
                        catalog = valid_ids if order == "original" else list(reversed(valid_ids))
                        base = search_command(question["text"], focus[question["id"]])
                        command = base + "\nValid original paragraph IDs: " + ", ".join(catalog)
                        row.update(status="running", command=command, catalog=catalog, command_body_sha256=fingerprint(base),
                                   handle=host.handle, prefix_sha256=fingerprint(host.payloads[method].prefix_ids))
                        call_started = perf_counter()
                        try:
                            generation = host.infer(method, command, budget=96)
                        except Exception as exc:
                            row.update(status="model_failed", error={"type": type(exc).__name__, "message": str(exc)},
                                       observed_infer_wall_seconds=perf_counter()-call_started)
                            emit("case_failed", case=row)
                            raise
                        row.update(generation=generation, observed_infer_wall_seconds=perf_counter()-call_started,
                                   observation=catalog_observation(generation["output"], catalog))
                        if order == "original":
                            old = reference["original_cases"][method+"/"+question["id"]]
                            if any(fingerprint(generation[k]) != old[k] for k in ("command", "input_ids")):
                                raise RuntimeError("Original command/input differs from the frozen run")
                            row["previous_output_tokens_reproduced"] = fingerprint(generation["generated_token_ids"]) == old["generated_token_ids"]
                        try:
                            value = validate_output(generation, {"kind": "locate", "valid_ids": valid_ids})
                            row.update(status="completed", contract_pass=True, value=value)
                        except Exception as exc:
                            row.update(status="contract_failed", error={"type": type(exc).__name__, "message": str(exc)})
                        emit("diagnostic_case", case=row)
                        print(paper["id"], question["id"], method, order, row["status"], flush=True)
                    before = perf_counter()
                    audit = host.audit()
                    audit.update(paper_id=paper["id"], question_id=question["id"], method=method, seconds=perf_counter()-before)
                    result["audits"].append(audit)
                    emit("context_audit", audit=audit)
        result["status"] = "completed"
    except Exception as exc:
        result.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        emit("run_error", error=result["error"])
        raise
    finally:
        result.update(observed_batch_wall_seconds=perf_counter()-started, event_logging_seconds=logs,
                      timing_scope="one model load, actual planning/handoff/prefill/fusion, 24 direct model requests and integrity audits; no OS workers or answers, one repeat, not a system speed benchmark")
        with (output_dir / "results.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
        emit("run_finished", status=result["status"])
        log.close()
