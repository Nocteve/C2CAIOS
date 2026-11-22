"""Single GPU owner. Actual planning history is immutable and versioned."""
from time import perf_counter
from .records import fingerprint
from .workload import (SYSTEM, plan_request, plan_schema, parse_plan, brief_command,
                       document, reference_address_audit)

METHODS = ("text_full", "text_brief", "c2c_shared")
MAP = dict(zip(METHODS, ("text_full_cached", "text_brief_cached", "c2c_shared")))


class PlanningContractError(ValueError):
    """A real model output cannot be dispatched; it remains a quality failure."""


class ModelHost:
    def __init__(self, config, lock):
        from .loading import load_models
        import torch
        torch.set_num_threads(4)
        torch.manual_seed(config["seed"])
        self.config, self.lock = config, lock
        self.combined, self.tokenizers = load_models(config, lock)
        self.combined.multi_source_fusion_mode = "sequential"
        self.receiver, self.sender = self.combined.model_list
        self.payloads = {}
        self.leases = 0

    def constraints(self, tokenizer, schema):
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
        return build_transformers_prefix_allowed_tokens_fn(tokenizer, JsonSchemaParser(schema))

    def prepare(self, paper, emit, *, mechanism=False, order="paper_first"):
        from .transcript import capture_plan
        from .cache_ops import prefill, cache_digest, cache_bytes
        from .kv import PlanningPayload, fuse_context
        from .semantic_scope import planning_scope
        self.payloads.clear()
        started = perf_counter()
        t = capture_plan(self.sender, self.tokenizers["sender"], self.tokenizers["receiver"],
            plan_request(paper, order), 320, self.config["max_input_tokens"],
            allowed_tokens=self.constraints(self.tokenizers["sender"], plan_schema(paper)))
        self.planning_seconds = perf_counter() - started
        emit("planning", output=t.output, prompt_ids=t.prompt_ids, generated_ids=t.generated_ids,
             prefix_ids=t.prefix_ids, prompt_text=t.prompt_text, metrics=t.metrics, seconds=self.planning_seconds)
        if not t.metrics["complete_assistant_turn"]:
            raise PlanningContractError("Actual Planner turn did not finish; no fabricated plan or context")
        parse_started = perf_counter()
        try:
            plan = parse_plan(t.output, paper)
        except (ValueError, TypeError, KeyError) as exc:
            raise PlanningContractError(str(exc)) from exc
        self.planning_seconds += perf_counter() - parse_started
        scope_started = perf_counter()
        semantic_indices, semantic_metadata = planning_scope(t, paper, self.tokenizers["receiver"])
        scope_seconds = perf_counter() - scope_started
        self.prefix, self.source = t.prefix_ids, t.source_cache
        self.native, native_seconds = prefill(self.receiver, self.prefix)
        fusion_timings = {}
        fused, fusion_seconds = fuse_context(self.combined, self.source, self.native,
                                             token_indices=semantic_indices, timings=fusion_timings)
        self.payloads = {
            "text_full": PlanningPayload(tuple(self.prefix), self.native),
            "c2c_shared": PlanningPayload(tuple(self.prefix), fused, fused=True, semantic_token_indices=semantic_indices)}
        self.handle = paper["id"] + ":" + fingerprint(self.prefix)[:16]
        audit_started = perf_counter()
        self.hashes = {"source": cache_digest(self.source), "native": cache_digest(self.native),
                       "fused": cache_digest(fused)}
        reference_audits = {
            "text_full": reference_address_audit(paper, document(paper), origin="full source document in native prefix"),
            "c2c_shared": reference_address_audit(paper, document(paper), origin="document in actual Planner source; receiver use unverified")}
        self.preparation_metrics = {"order": order, "planning": t.metrics,
            "planning_seconds": self.planning_seconds, "receiver_prefix_prefill_seconds": native_seconds,
            "scope_selection_seconds": scope_seconds, "fusion": fusion_timings,
            "integrity_audit_seconds": perf_counter()-audit_started,
            "source_cache_bytes": cache_bytes(self.source), "receiver_prefix_cache_bytes": cache_bytes(self.native),
            "fused_cache_bytes": cache_bytes(fused), "prefix_tokens": len(self.prefix),
            "kv_bytes_are_not_network_traffic": True, "scorer_training_or_inference": False}
        self.preparation_metrics["reference_address_audit"] = reference_audits
        self.costs = {"text_full": native_seconds, "c2c_shared": native_seconds + fusion_seconds + scope_seconds}
        self.semantic_metadata = semantic_metadata
        self.scope_seconds = scope_seconds
        if mechanism:
            emit("publication", handle=self.handle, actual_planner_kv=True, source_reencoded=False,
                 includes_minimal_dispatch_history=True, tokens=len(self.prefix), hashes=self.hashes,
                 costs=self.costs, physical_zero_copy=False, planning_seconds=self.planning_seconds,
                 semantic_scope=semantic_metadata, semantic_token_indices=list(semantic_indices),
                 scope_selection_seconds=scope_seconds, text_handoff_generated=False)
            emit("reference_address_audit", methods=reference_audits)
            return plan
        started = perf_counter()
        brief = capture_plan(self.sender, self.tokenizers["sender"], self.tokenizers["receiver"],
            [{"role": "user", "content": brief_command(paper)}], 1600, self.config["max_input_tokens"],
            producer_role="text handoff", prior_prefix_ids=self.prefix, prior_cache=self.source)
        self.costs["text_brief"] = perf_counter() - started
        self.preparation_metrics["brief_generation_seconds"] = self.costs["text_brief"]
        self.preparation_metrics["brief"] = brief.metrics
        emit("brief", output=brief.output, prompt_ids=brief.prompt_ids, generated_ids=brief.generated_ids, metrics=brief.metrics,
             seconds=self.costs["text_brief"])
        audit_started = perf_counter()
        reference_audits["text_brief"] = reference_address_audit(
            paper, brief.output, origin="actual generated text handoff")
        self.preparation_metrics["integrity_audit_seconds"] += perf_counter()-audit_started
        if brief.metrics["complete_assistant_turn"] and brief.output.strip():
            started = perf_counter()
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": brief.output}]
            ids = self.tokenizers["receiver"].apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
            cache, seconds = prefill(self.receiver, ids)
            self.payloads["text_brief"] = PlanningPayload(tuple(ids), cache)
            receiver_brief_seconds = perf_counter()-started
            self.costs["text_brief"] += receiver_brief_seconds
            self.preparation_metrics.update(brief_receiver_prefill_seconds=seconds,
                brief_receiver_setup_seconds=receiver_brief_seconds, brief_prefix_tokens=len(ids),
                brief_cache_bytes=cache_bytes(cache))
            audit_started = perf_counter()
            self.hashes["brief"] = cache_digest(cache)
            self.preparation_metrics["integrity_audit_seconds"] += perf_counter()-audit_started
        emit("publication", handle=self.handle, actual_planner_kv=True, source_reencoded=False,
             includes_minimal_dispatch_history=True, tokens=len(self.prefix), hashes=self.hashes,
             costs=self.costs, physical_zero_copy=False, planning_seconds=self.planning_seconds,
             semantic_scope=semantic_metadata, semantic_token_indices=list(semantic_indices),
             scope_selection_seconds=scope_seconds, preparation_metrics=self.preparation_metrics)
        return plan

    def infer(self, method, command, budget=128, context="paper"):
        from .transcript import worker_ids
        from .kv import generate_worker
        from .cache_ops import prefill, decode
        self.leases += 1
        try:
            if context == "evidence_only":
                # New clean context: no paper cache and no Planner transcript.
                import torch
                allocated_before = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                tok = self.tokenizers["receiver"]
                ids = tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": command}], tokenize=True,
                    add_generation_prompt=True, enable_thinking=False)
                if len(ids) + budget > self.config["max_input_tokens"]:
                    raise ValueError("Evidence-only request exceeds context limit; no truncation")
                cache, seconds = prefill(self.receiver, ids[:-1])
                prefill_peak = torch.cuda.max_memory_allocated()
                result = decode(self.receiver, tok, ids, ids[:-1], cache, budget)
                result["peak_allocated_mib"] = max(prefill_peak / 2**20, result["peak_allocated_mib"])
                result["allocated_before_mib"] = allocated_before / 2**20
                result["peak_incremental_mib"] = result["peak_allocated_mib"] - result["allocated_before_mib"]
                result["service_seconds"] += seconds
                result["first_token_seconds"] += seconds
                result["receiver_task_prefill_seconds"] = seconds + result["clone_seconds"]
                result["task_clone_seconds"] = result["clone_seconds"]
                result["task_prefill_forward_seconds"] = result["receiver_task_prefill_seconds"] - result["clone_seconds"]
                result["context_authority"] = "retrieved artifact only; no full-paper cache"
                return {**result, "input_ids": ids, "command": command}
            payload = self.payloads[method]
            ids = worker_ids(self.tokenizers["receiver"], list(payload.prefix_ids), command,
                             self.config["max_input_tokens"])
            if len(ids) + budget > self.config["max_input_tokens"]:
                raise ValueError("Evidence search exceeds context limit; no truncation")
            result = generate_worker(self.combined, self.tokenizers, ids, payload, MAP[method], budget)
            return {**result, "input_ids": ids, "command": command, "handle": self.handle,
                    "context_authority": "paper communication context"}
        finally:
            self.leases -= 1

    def audit(self):
        from .cache_ops import cache_digest
        hashes = {"source": cache_digest(self.source), "native": cache_digest(self.native),
                  "fused": cache_digest(self.payloads["c2c_shared"].receiver_cache)}
        if "text_brief" in self.payloads:
            hashes["brief"] = cache_digest(self.payloads["text_brief"].receiver_cache)
        if hashes != self.hashes or self.leases:
            raise RuntimeError("Published KV mutated or live leases remain")
        return {"unchanged": True, "active_leases": self.leases}
