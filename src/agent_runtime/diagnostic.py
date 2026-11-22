"""Bounded development diagnosis, invoked only through scripts/run.py."""
from time import perf_counter

from .runtime import validate_output
from .workload import search_command

CONDITIONS = ("native", "paper", "focus", "paper_focus")


def expected_cases(papers):
    return [{"paper_id": p["id"], "question_id": q["id"], "condition": condition,
             "status": "not_run", "contract_pass": False, "content_takeovers": 0}
            for p in papers for q in p["questions"][:2] for condition in CONDITIONS]


def source_cases(papers):
    return [{"paper_id": p["id"], "question_id": q["id"], "condition": "source_4b",
             "status": "not_run", "contract_pass": False, "content_takeovers": 0}
            for p in papers for q in p["questions"][:2]]


def run_source_cases(host, paper, plan, cases, emit):
    """Four source-side requests, kept out of Worker communication scores."""
    from .cache_ops import prefill, decode
    from .transcript import worker_ids
    requests = commands(paper, plan)
    for row in [r for r in cases if r["paper_id"] == paper["id"]]:
        ids = worker_ids(host.tokenizers["sender"], host.prefix, requests[row["question_id"]],
                         host.config["max_input_tokens"])
        if len(ids) + 96 > host.config["max_input_tokens"]:
            raise ValueError("Source diagnostic exceeds context limit; no truncation")
        row.update(command=requests[row["question_id"]], budget=96, input_ids=ids,
                   source_sha256=host.hashes["source"], handle=host.handle, status="running",
                   producer="4B source diagnostic only; not delivered to Workers")
        host.leases += 1
        try:
            cache, seconds = prefill(host.sender, ids[:-1], host.source, len(host.prefix))
            generation = decode(host.sender, host.tokenizers["sender"], ids, ids[:-1], cache, 96)
            generation["service_seconds"] += seconds
            generation["first_token_seconds"] += seconds
            generation.update(source_task_prefill_seconds=seconds, source_history_reencoded=False,
                              input_ids=ids, command=row["command"], context_authority="actual source Planner cache")
            del cache
            row["generation"] = generation
            try:
                value = validate_output(generation, {"kind": "locate", "valid_ids": [p["id"] for p in paper["paragraphs"]]})
                row.update(value=value, contract_pass=True, status="completed")
            except Exception as exc:
                row.update(status="contract_failed", error={"type": type(exc).__name__, "message": str(exc)})
        except Exception as exc:
            row.update(status="model_failed", error={"type": type(exc).__name__, "message": str(exc)})
            raise
        finally:
            host.leases -= 1
            emit("source_diagnostic_case", **row)
        print(paper["id"], row["question_id"], "source_4b", row["status"], flush=True)
    emit("context_audit", **host.audit())


def commands(paper, plan):
    focus = {t["question_id"]: t["focus"] for t in plan}
    valid_ids = [p["id"] for p in paper["paragraphs"]]
    return {q["id"]: search_command(q["text"], focus[q["id"]])
            + "\nValid original paragraph IDs: " + ", ".join(valid_ids)
            for q in paper["questions"][:2]}


def numerical_audit(host, command):
    """Run unmodified upstream forward; reconstruct history ONLY for this audit.

    Both sides prefill prefix, then suffix[:-1], then the last input token.
    The frozen fuser is dense here; scope restoration is checked separately.
    This is not an equivalence claim for autoregressive vs teacher-forced KV.
    """
    import torch
    from .cache_ops import clone_cache, cache_digest, prefill, decode
    from .kv import fuse_context, PlanningPayload, generate_worker
    from .transcript import worker_ids

    def difference(a, b):
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError("Numerical audit shape/dtype mismatch")
        d = (a.float() - b.float()).abs()
        return {"equal": bool(torch.equal(a, b)), "max_abs": d.max().item(),
                "mean_abs": d.mean().item()}

    def cache_difference(a, b, length=None):
        if len(a) != len(b):
            raise ValueError("Numerical audit layer mismatch")
        rows = [{"layer": layer, "kv": name, **difference(x[:, :, :length], y[:, :, :length])}
                for layer, (pa, pb) in enumerate(zip(a, b))
                for name, x, y in zip(("key", "value"), pa, pb)]
        return {"equal": all(r["equal"] for r in rows),
                "max_abs": max(r["max_abs"] for r in rows), "layers": rows}

    def seed_logits(cache):
        return host.receiver(input_ids=ids_tensor[:, -1:], past_key_values=clone_cache(cache),
            attention_mask=mask, position_ids=positions[:, -1:], use_cache=True,
            logits_to_keep=1).logits[:, -1].clone()

    combined = host.combined
    if combined.include_response:
        raise ValueError("This audit requires upstream prefix-only fusion")
    ids = worker_ids(host.tokenizers["receiver"], host.prefix, command, host.config["max_input_tokens"])
    length = len(host.prefix)
    if len(ids) <= length + 1:
        raise ValueError("Audit requires a nonempty native command suffix")
    ids_tensor = torch.tensor([ids], device="cuda:0")
    mask = torch.ones_like(ids_tensor)
    positions = torch.arange(len(ids), device="cuda:0")[None]
    sections = [torch.ones((1, length, 2), device="cuda:0", dtype=torch.long),
                -torch.ones((1, len(ids) - length - 1, 2), device="cuda:0", dtype=torch.long)]
    started = perf_counter()
    report = {"protocol": "official prefix fusion + native suffix[:-1] + native last token",
              "scope": "one fixed first-paper first-question dense numerical fixture",
              "source_history": "teacher-forced reconstruction for numerical audit ONLY",
              "decoder_origin": "official forward cache + project cache_ops.decode versus project generate_worker; RosettaModel.generate not invoked",
              "actual_planner_kv_claim": False, "input_ids": ids, "command": command,
              "sections": [length, len(ids) - length - 1, 1],
              "masks": [1, -1, -1], "scope_masks_covered_by_upstream": False}
    try:
        with torch.inference_mode():
            def official():
                return combined(kv_cache_index=sections, input_ids=ids_tensor[:, :-1],
                    attention_mask=mask[:, :-1], position_ids=positions[:, :-1],
                    use_cache=True, logits_to_keep=1).past_key_values

            upstream = official()
            upstream_source = clone_cache(combined.kv_cache_dict[0][1])
            upstream_native = clone_cache(combined.kv_cache_dict[0][0])
            first = seed_logits(upstream)
            repeated = official()
            repeated_logits = seed_logits(repeated)
            report["repeat_cache"] = cache_difference(upstream, repeated)
            report["repeat_logits"] = difference(first, repeated_logits)
            del repeated, repeated_logits
            native, _ = prefill(host.receiver, host.prefix)
            source, _ = prefill(host.sender, host.prefix)
            report["native_prefill"] = cache_difference(upstream_native, native)
            report["source_prefill"] = cache_difference(upstream_source, source)
            report["source_sha256"] = cache_digest(source)
            report["native_sha256"] = cache_digest(native)
            fused, _ = fuse_context(combined, source, native)
            report["dense_fusion"] = cache_difference(upstream, fused, length)
            branch, _ = prefill(host.receiver, ids[:-1], fused, length)
            report["command_cache"] = cache_difference(upstream, branch)
            adapted_logits = seed_logits(branch)
            report["first_logits"] = difference(first, adapted_logits)
            report["first_token_ids"] = [first.argmax(-1).item(), adapted_logits.argmax(-1).item()]
            official_output = decode(host.receiver, host.tokenizers["receiver"], ids, ids[:-1], upstream, 96)
            adapted_output = generate_worker(combined, host.tokenizers, ids,
                PlanningPayload(tuple(host.prefix), fused, fused=True), "c2c_shared", 96)
            report["upstream_cache_local_decode"] = official_output
            report["adapter_generation"] = adapted_output
            report["same_generated_ids"] = official_output["generated_token_ids"] == adapted_output["generated_token_ids"]
            report["projectors"] = [{"index": i, "class": type(p).__name__,
                "parameters": sum(v.numel() for v in p.parameters()),
                "key_gate_logit": float(p.key_gate_logit), "value_gate_logit": float(p.value_gate_logit)}
                for i, p in enumerate(combined.projector_list)]
            report["routing"] = combined.projector_dict
            report["passed"] = all(report[k]["equal"] for k in
                ("repeat_cache", "repeat_logits", "native_prefill", "source_prefill", "dense_fusion", "command_cache", "first_logits")) and report["same_generated_ids"]
            # Exact equality is a conservative stop rule. A mismatch is retained
            # for diagnosis, not automatically called an upstream implementation bug.
            report["acceptance"] = "exact equality on this fixed fixture; otherwise stop and diagnose"
    finally:
        combined.kv_cache_dict = {}
    report["seconds_including_integrity_work"] = perf_counter() - started
    return report


def run_cases(host, paper, plan, cases, emit):
    """16 retrievals across two calls. No OS worker, repair, handoff or answer."""
    import torch
    from .cache_ops import cache_digest
    from .kv import PlanningPayload, fuse_context
    requests = commands(paper, plan)
    original = host.payloads["c2c_shared"]
    scope = host.semantic_metadata
    try:
        for condition in CONDITIONS:
            selected = None
            fusion_seconds = 0.0
            if condition == "native":
                payload, method = host.payloads["text_full"], "text_full"
            else:
                selected = tuple(scope["paper_indices"] if condition == "paper" else
                    scope["actual_focus_indices"] if condition == "focus" else original.semantic_token_indices)
                if condition == "paper_focus":
                    payload = original
                    fusion_seconds = host.costs["c2c_shared"] - host.costs["text_full"] - host.scope_seconds
                else:
                    cache, fusion_seconds = fuse_context(host.combined, host.source, host.native, token_indices=selected)
                    payload = PlanningPayload(tuple(host.prefix), cache, fused=True, semantic_token_indices=selected)
                host.payloads["c2c_shared"] = payload
                method = "c2c_shared"
            # Validate exact restoration without interpreting excluded positions as
            # purely control. Source remains the real, immutable Planner cache.
            excluded = sorted(set(range(len(host.prefix))) - set(selected or ()))
            keep = torch.tensor(excluded, device="cuda:0", dtype=torch.long)
            with torch.inference_mode():
                preserved = all(torch.equal(x.index_select(2, keep), y.index_select(2, keep))
                                for a, b in zip(payload.receiver_cache, host.native) for x, y in zip(a, b))
                changed = sum(int(torch.count_nonzero(x != y)) for a, b in zip(payload.receiver_cache, host.native) for x, y in zip(a, b))
            if not preserved:
                raise RuntimeError("Scope restoration changed excluded native positions")
            payload_hash = cache_digest(payload.receiver_cache)
            emit("diagnostic_condition", condition=condition, selected_indices=list(selected or ()),
                 source_sha256=host.hashes["source"], native_sha256=host.hashes["native"],
                 payload_sha256=payload_hash, excluded_exactly_native=preserved,
                 changed_tensor_elements=changed, fusion_seconds=fusion_seconds,
                 projection_is_dense=condition != "native", scoring_trained=False)
            for question in paper["questions"][:2]:
                row = next(r for r in cases if r["paper_id"] == paper["id"] and r["question_id"] == question["id"] and r["condition"] == condition)
                row.update(command=requests[question["id"]], budget=96, payload_sha256=payload_hash,
                           source_sha256=host.hashes["source"], native_sha256=host.hashes["native"],
                           handle=host.handle, status="running")
                try:
                    generation = host.infer(method, row["command"], budget=96)
                    row["generation"] = generation
                    try:
                        value = validate_output(generation, {"kind": "locate", "valid_ids": [p["id"] for p in paper["paragraphs"]]})
                        row.update(value=value, contract_pass=True, status="completed")
                    except Exception as exc:
                        row.update(status="contract_failed", error={"type": type(exc).__name__, "message": str(exc)})
                except Exception as exc:
                    row.update(status="model_failed", error={"type": type(exc).__name__, "message": str(exc)})
                    raise
                finally:
                    emit("diagnostic_case", **row)
                print(paper["id"], question["id"], condition, row["status"], flush=True)
            if cache_digest(payload.receiver_cache) != payload_hash:
                raise RuntimeError("Diagnostic payload was mutated")
    finally:
        host.payloads["c2c_shared"] = original
    emit("context_audit", **host.audit())


def run_arc_native_calibration(output_dir):
    """Fill one missing ablation using public exact inputs; no online answer key."""
    import hashlib
    import json
    import os
    from pathlib import Path
    from .project import ROOT
    from .loading import load_models
    from .cache_ops import prefill, decode, cache_digest
    import torch
    import transformers

    data=ROOT/'data/arc-native-calibration'
    manifest=json.loads((data/'manifest.json').read_text())
    raw=(data/'public.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=manifest['public_sha256']:
        raise ValueError('Frozen diagnostic input changed')
    cases=json.loads(raw)
    if [c['id'] for c in cases]!=manifest['expected_ids']:
        raise ValueError('Frozen diagnostic denominator changed')
    lock=json.loads((ROOT/'configs/cloud.models.lock.json').read_text())
    if lock!=manifest['models']:
        raise ValueError('Historical and current model pins differ')
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=False)
    started=perf_counter();torch.set_num_threads(4);torch.manual_seed(manifest['config']['seed'])
    result={'status':'running','pid':os.getpid(),'manifest':manifest,'reference_answers_read':False,
            'scope':'Native receiver ablation only, not a new main baseline or autonomous workflow; no cross-run speedup claim',
            'model_residency':'both pinned models and fuser loaded as in historical calibration; only receiver used',
            'c2c_executed':False,'worker_output_constraints':False,
            'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [*sorted((ROOT/'src/agent_runtime').glob('*.py')),ROOT/'scripts/run.py']},
            'cases':[{'id':c['id'],'status':'not_run'} for c in cases]}
    try:
        load_at=perf_counter();combined,toks=load_models(manifest['config'],lock)
        result.update(loading_seconds=perf_counter()-load_at,torch=str(torch.__version__),
                      transformers=transformers.__version__,gpu=torch.cuda.get_device_name())
        receiver=combined.model_list[0];tok=toks['receiver']
        for i,case in enumerate(cases):
            row=result['cases'][i];cache=None;began=perf_counter()
            try:
                if set(case)!={'id','input_ids','receiver_content','decode_boundary','historical_native_cache_sha256'}:
                    raise ValueError('Unexpected public input fields')
                ids=tok.apply_chat_template([{'role':'user','content':case['receiver_content']}],
                    tokenize=True,add_generation_prompt=True,enable_thinking=False)
                if ids!=case['input_ids'] or case['decode_boundary']!=len(ids)-1:
                    raise ValueError('Original exact input and L-1/1 boundary do not match')
                cache,seconds=prefill(receiver,ids[:-1]);audit_at=perf_counter();before=cache_digest(cache)
                audit_seconds=perf_counter()-audit_at
                generated=decode(receiver,tok,ids,ids[:-1],cache,max_new_tokens=manifest['output_tokens'])
                audit_at=perf_counter();after=cache_digest(cache);audit_seconds+=perf_counter()-audit_at
                row.update(status='generated',input_ids=ids,decode_boundary=len(ids)-1,generation=generated,
                    native_prefill_seconds=seconds,cache_sha256=before,cache_unchanged=before==after,
                    historical_native_cache_matches=before==case['historical_native_cache_sha256'],
                    integrity_audit_seconds=audit_seconds,
                    inference_seconds_excluding_integrity_audit=seconds+generated['service_seconds'])
                if before!=after:raise RuntimeError('Native input cache mutated')
            except Exception as exc:
                row.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)})
                if isinstance(exc,RuntimeError):raise
            finally:
                row['observed_case_wall_seconds']=perf_counter()-began
                (output/(case['id']+'.json')).write_text(json.dumps(row,indent=2),encoding='utf-8')
                del cache
            if (i+1)%16==0:print('native receiver',i+1,'/',len(cases),flush=True)
        result['status']='completed'
    except Exception as exc:
        result.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)})
        raise
    finally:
        result['observed_batch_wall_seconds']=perf_counter()-started
        (output/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
