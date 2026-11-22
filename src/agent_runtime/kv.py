"""Private Worker branches from a published, actual Planner transcript.

The caller captures Planner state with ``transcript.capture_plan`` and owns its
publication/leases through ModelHost. Frozen payload fields do not freeze
their tensors: every forward here branches before extending a published cache.
Only the pinned single-GPU, single-Sharer, batch-one setup is supported.
"""

from dataclasses import dataclass
from time import perf_counter

import torch

from .cache_ops import clone_cache, prefill


METHODS = frozenset({"text_full_cached", "text_brief_cached", "c2c_shared", "c2c_hidden", "c2c_command"})


def _token_ids(ids, name):
    if not isinstance(ids, (list, tuple)) or not ids:
        raise ValueError(f"{name} must be a nonempty list or tuple of token IDs")
    if any(type(token) is not int or token < 0 for token in ids):
        raise ValueError(f"{name} contains an invalid token ID")


def _check_cache(cache, length, name):
    if cache is None or not hasattr(cache, "get_seq_length"):
        raise ValueError(f"{name} must be a populated Transformer cache")
    if cache.get_seq_length() != length or len(cache) == 0:
        raise ValueError(f"{name} does not cover the exact published token prefix")
    for pair in cache:
        if len(pair) != 2:
            raise ValueError(f"{name} contains a malformed key/value pair")
        key, value = pair
        if not all(isinstance(tensor, torch.Tensor) and tensor.ndim == 4 for tensor in pair):
            raise ValueError(f"{name} requires four-dimensional KV tensors")
        if key.shape != value.shape or key.shape[0] != 1 or key.shape[2] != length or min(key.shape) < 1:
            raise ValueError(f"{name} requires aligned, nonempty batch-one KV tensors")
        if key.dtype != value.dtype or key.device != value.device or not key.is_floating_point():
            raise ValueError(f"{name} has inconsistent KV precision or device")


@dataclass(frozen=True)
class PlanningPayload:
    """Cache ownership references plus exact token identity at publication.

``receiver_cache`` is native unless ``fused`` is true.  ``nbytes`` counts unique
retained receiver tensor storages, including any backing-view storage;
it excludes model weights, Python objects, and the CUDA allocator's reservation.
The caller remains responsible for model/fuser identity and cache fingerprints.
"""

    prefix_ids: tuple[int, ...]
    receiver_cache: object
    fused: bool = False
    semantic_token_indices: tuple[int, ...] | None = None
    hidden_context_tokens: int = 0

    def __post_init__(self):
        if not isinstance(self.prefix_ids, tuple):
            raise ValueError("Published prefix_ids must be an immutable tuple")
        _token_ids(self.prefix_ids, "prefix_ids")
        if type(self.fused) is not bool:
            raise ValueError("fused must be an explicit boolean")
        if type(self.hidden_context_tokens) is not int or self.hidden_context_tokens < 0:
            raise ValueError("hidden_context_tokens must be a nonnegative integer")
        _check_cache(self.receiver_cache, len(self.prefix_ids) + self.hidden_context_tokens, "receiver_cache")
        if self.semantic_token_indices is not None:
            _validate_indices(self.semantic_token_indices, len(self.prefix_ids))

    @property
    def nbytes(self):
        storages = {}
        for pair in self.receiver_cache:
            for tensor in pair:
                storage = tensor.untyped_storage()
                identity = (str(tensor.device), storage.data_ptr())
                storages[identity] = storage.nbytes()
        return sum(storages.values())


def _validate_indices(indices, length):
    if (not isinstance(indices, tuple) or not indices or any(type(i) is not int for i in indices)
            or tuple(sorted(set(indices))) != indices or indices[0] < 0 or indices[-1] >= length):
        raise ValueError("Fusion positions must be sorted unique actual cache indices")


def _fusion_mode(combined, requested=None):
    configured = getattr(combined, "multi_source_fusion_mode", None)
    if configured not in {"sequential", "parallel"}:
        raise ValueError("Configure an explicit sequential or parallel fusion mode")
    if requested is not None and requested != configured:
        raise ValueError("Requested fusion arithmetic differs from the configured upstream mode")
    return configured


def _models(combined):
    if getattr(combined, "base_model_idx", None) != 0 or len(combined.model_list) != 2:
        raise ValueError("This runtime requires Receiver 0 and exactly one Sharer 1")
    receiver, source = combined.model_list
    if any(model.training for model in (combined, receiver, source)):
        raise ValueError("Published frozen-model inference requires eval mode")
    if any(torch.device(model.device) != torch.device("cuda:0") for model in (receiver, source)):
        raise ValueError("The existing prefill adapter requires both models on cuda:0")
    return receiver, source


def _cache_for_model(cache, model, name):
    if len(cache) != model.config.num_hidden_layers:
        raise ValueError(f"{name} layer count does not match its model")
    if any(tensor.device != model.device or tensor.dtype != model.dtype for pair in cache for tensor in pair):
        raise ValueError(f"{name} does not match the loaded model device and precision")


@torch.inference_mode()
def fuse_context(combined, source, receiver, *, fusion_mode=None, token_indices=None, timings=None):
    """Return ``(private_fused_cache, seconds)`` using explicit upstream arithmetic.

The configured ``combined.multi_source_fusion_mode`` is authoritative.  For one
source, sequential copies the projected result; parallel evaluates the distinct
expression ``original + (projected - original)``.  Upstream eval gates use FP32,
so projection may promote BF16 inputs to FP32.  Preserve that promoted arithmetic
until copy_ converts to the target cache dtype, matching upstream assignment;
casting projected tensors earlier would change the parallel expression.
Neither input is written.  Timing includes validation, cloning, projection, and
completion synchronization.
"""
    torch.cuda.synchronize()
    started = perf_counter()
    mode = _fusion_mode(combined, fusion_mode)
    receiver_model, source_model = _models(combined)
    if receiver is None or not hasattr(receiver, "get_seq_length"):
        raise ValueError("Fusion requires a populated receiver cache")
    length = receiver.get_seq_length()
    _check_cache(receiver, length, "receiver_cache")
    _check_cache(source, length, "source_cache")
    if token_indices is not None:
        _validate_indices(token_indices, length)
    _cache_for_model(receiver, receiver_model, "receiver_cache")
    _cache_for_model(source, source_model, "source_cache")
    routes = combined.projector_dict.get(0, {})
    if set(routes) != {1} or not routes[1]:
        raise ValueError("Fusion requires exactly the pinned single-source mapping")
    for target_layer, pairs in routes[1].items():
        if type(target_layer) is not int or not 0 <= target_layer < len(receiver) or len(pairs) != 1:
            raise ValueError("Fusion requires one valid projector per mapped receiver layer")
        source_layer, projector_index = pairs[0]
        if (type(source_layer) is not int or not 0 <= source_layer < len(source)
                or type(projector_index) is not int or not 0 <= projector_index < len(combined.projector_list)):
            raise ValueError("Invalid source-layer or projector index")
        if combined.projector_list[projector_index].training:
            raise ValueError("Fusion projectors must be in eval mode")
    try:
        validated = perf_counter()
        fused = clone_cache(receiver)
        torch.cuda.synchronize()
        cloned = perf_counter()
        for target_layer, pairs in routes[1].items():
            source_layer, projector_index = pairs[0]
            original = receiver[target_layer]
            projected = combined.projector_list[projector_index](source[source_layer], original)
            if not isinstance(projected, (tuple, list)) or len(projected) != 2:
                raise RuntimeError("Projector did not return one key/value pair")
            for output_tensor, original_tensor, projected_tensor in zip(fused[target_layer], original, projected):
                if (not isinstance(projected_tensor, torch.Tensor)
                        or projected_tensor.shape != original_tensor.shape
                        or not projected_tensor.is_floating_point()
                        or projected_tensor.device != original_tensor.device):
                    raise RuntimeError("Projector must return floating KV on the receiver shape and device")
                output_tensor.copy_(projected_tensor if mode == "sequential"
                                    else original_tensor + (projected_tensor - original_tensor))
        torch.cuda.synchronize()
        projected_at = perf_counter()
        if token_indices is not None:
            keep = torch.zeros(length, dtype=torch.bool, device=receiver_model.device)
            keep[list(token_indices)] = True
            excluded = (~keep).nonzero(as_tuple=True)[0]
            # Projection is still dense. Restore all excluded positions after
            # projection; this policy does NOT claim sparse compute savings.
            for target, native in zip(fused, receiver):
                for tensor, original_tensor in zip(target, native):
                    tensor.index_copy_(2, excluded, original_tensor.index_select(2, excluded))
    finally:
        torch.cuda.synchronize()
    finished = perf_counter()
    if timings is not None:
        timings.update(validation_seconds=validated-started, clone_seconds=cloned-validated,
                       projection_seconds=projected_at-cloned, restore_seconds=finished-projected_at,
                       total_seconds=finished-started)
    return fused, finished - started


def _extend_private(model, ids, published_cache, prefix_length, timings):
    """Extend a clone; the zero-tail case must still produce private ownership."""
    if len(ids) > prefix_length:
        return prefill(model, ids, published_cache, prefix_length, timings=timings)
    torch.cuda.synchronize()
    started = perf_counter()
    branch = clone_cache(published_cache)
    torch.cuda.synchronize()
    timings.update(clone_seconds=perf_counter()-started)
    return branch, perf_counter() - started


@torch.inference_mode()
def generate_worker(combined, tokenizers, full_ids, payload, method, max_new_tokens, *, sampling=None, sample_seed=None):
    """Generate from one leased payload, returning generation_record-compatible data.

All methods split at ``L-1`` and process the last input token before greedy
decoding. The explicit current command extends the published receiver context
natively; it is not fused with a new source task cache. No history is re-tokenized or
re-prefilled, no source decoding occurs, and no post-task cache is published.
Caller checks context budgets and keeps the lease through this function's final
CUDA synchronization, including exceptional exits.
"""
    if method not in METHODS or type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("Invalid Worker method or output budget")
    if not isinstance(payload, PlanningPayload):
        raise TypeError("Worker requires a PlanningPayload")
    _token_ids(full_ids, "full_ids")
    if tuple(full_ids[:len(payload.prefix_ids)]) != payload.prefix_ids or len(full_ids) <= len(payload.prefix_ids):
        raise ValueError("Worker input must extend the exact published prefix")
    if payload.fused != (method in {"c2c_shared", "c2c_hidden", "c2c_command"}):
        raise ValueError("Only c2c_shared/c2c_hidden/c2c_command may consume an already fused published cache")
    receiver, _ = _models(combined)
    mode = _fusion_mode(combined)
    tokenizer = tokenizers["receiver"]
    ids = list(full_ids)
    prefix_length = len(payload.prefix_ids) + payload.hidden_context_tokens
    instruction_length = len(ids) - 1
    torch.cuda.synchronize()
    started = perf_counter()
    allocated_before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    validation_started = perf_counter()
    _check_cache(payload.receiver_cache, prefix_length, "published receiver cache")
    _cache_for_model(payload.receiver_cache, receiver, "published receiver cache")
    for token_id in set(ids):
        if tokenizer.convert_ids_to_tokens(token_id) is None:
            raise ValueError("Worker input contains an unknown receiver token")
    eos = receiver.generation_config.eos_token_id
    eos_ids = [] if eos is None else list(eos) if isinstance(eos, (list, tuple)) else [eos]
    if any(type(token) is not int or token < 0 for token in eos_ids):
        raise ValueError("Receiver generation configuration has invalid EOS IDs")
    validation_seconds = perf_counter() - validation_started
    selected = payload.semantic_token_indices
    warpers=[];generator=None
    if sampling is not None:
        from transformers import TemperatureLogitsWarper,TopKLogitsWarper,TopPLogitsWarper
        warpers=[TemperatureLogitsWarper(sampling['temperature']),TopKLogitsWarper(sampling['top_k']),TopPLogitsWarper(sampling['top_p'])]
        if type(sample_seed) is not int:raise ValueError('Sampling requires a recorded per-request seed')
        generator=torch.Generator(device='cuda:0').manual_seed(sample_seed)
    try:
        prefill_timings = {}
        branch, receiver_seconds = _extend_private(receiver, ids[:-1], payload.receiver_cache, prefix_length, prefill_timings)
        _check_cache(branch, instruction_length + payload.hidden_context_tokens, "private receiver instruction cache")
        decode_started = perf_counter()
        output = receiver(input_ids=torch.tensor([[ids[-1]]], device="cuda:0"),
                          attention_mask=torch.ones((1, len(ids) + payload.hidden_context_tokens), dtype=torch.long, device="cuda:0"),
                          position_ids=torch.tensor([[instruction_length + payload.hidden_context_tokens]], device="cuda:0"),
                          past_key_values=branch, use_cache=True, logits_to_keep=1)
        past, logits = output.past_key_values, output.logits[:, -1, :]
        del output, branch
        generated, first_token_seconds = [], None
        for _ in range(max_new_tokens):
            if warpers:
                scores=logits.float()
                for warper in warpers:scores=warper(None,scores)
                token=torch.multinomial(torch.softmax(scores,dim=-1),1,generator=generator).squeeze(-1)
            else:
                token = logits.argmax(dim=-1)
            generated.append(token.item())
            if first_token_seconds is None:
                first_token_seconds = perf_counter() - started
            if generated[-1] in eos_ids or len(generated) == max_new_tokens:
                break
            position = len(ids) + len(generated) - 1
            output = receiver(input_ids=token[:, None], past_key_values=past, use_cache=True,
                              attention_mask=torch.ones((1, position + 1 + payload.hidden_context_tokens), dtype=torch.long, device="cuda:0"),
                              position_ids=torch.tensor([[position + payload.hidden_context_tokens]], device="cuda:0"), logits_to_keep=1)
            past, logits = output.past_key_values, output.logits[:, -1, :]
            del output
        torch.cuda.synchronize()
        decode_seconds = perf_counter() - decode_started
        text = tokenizer.decode(generated, skip_special_tokens=True)
        del past, logits
    finally:
        torch.cuda.synchronize()
    elapsed = perf_counter() - started
    return {
        "method": method, "output": text, "generated_token_ids": generated,
        "output_tokens": len(generated), "prompt_tokens": len(ids),
        "cached_prefix_tokens": prefix_length,
        "stop_reason": "eos" if generated[-1] in eos_ids else "token_limit",
        "service_seconds": elapsed, "first_token_seconds": first_token_seconds,
        "input_validation_seconds": validation_seconds,
        "receiver_task_prefill_seconds": receiver_seconds,
        "task_clone_seconds": prefill_timings["clone_seconds"],
        "task_prefill_forward_seconds": receiver_seconds-prefill_timings["clone_seconds"],
        "task_prefill_timing_includes_cache_clone": True,
        "decode_seconds": decode_seconds, "decode_timing_includes_final_input_token": True,
        "runtime_overhead_seconds": elapsed - validation_seconds - receiver_seconds - decode_seconds,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "allocated_before_mib": allocated_before / 2**20,
        "peak_incremental_mib": (torch.cuda.max_memory_allocated()-allocated_before) / 2**20,
        "fusion_mode": mode if method.startswith("c2c_") else None,
        "published_cache_fused": payload.fused,
        "receiver_task_context": "published_fused_prefix" if payload.fused else "published_native_prefix",
          "source_history_reencoded": False,
          "sampling":sampling,"sample_seed":sample_seed,
        "published_prefix_token_range": [0, prefix_length],
        "hidden_context_tokens": payload.hidden_context_tokens,
        "receiver_task_prefill_token_range": [prefix_length, instruction_length],
        "fusion_token_range": [0, prefix_length] if method == "c2c_shared" else None,
        "fusion_selected_token_indices": list(selected) if selected is not None else None,
        "fusion_range_is_dense_projection_not_full_injection": selected is not None,
        "fusion_performed_this_request": False,
        "decode_seed_token_range": [instruction_length, len(ids)],
        "token_ranges_are_half_open": True,
        "position_policy": "contiguous absolute positions starting at zero; no history reset",
        "physical_prefix_zero_copy": False,
    }


def score_worker_continuation(combined, tokenizers, full_ids, payload, continuation_ids):
    """Teacher-forced sum-logprob of a fixed continuation under one published payload.

    Mirrors generate_worker's branch/position/attention conventions exactly;
    instead of greedy decoding, feeds continuation_ids and sums
    log p(target_t | prefix, targets_<t). Used by the A3 receiver-utility
    ablation: utility of a block = L(full selection) - L(drop-one selection).
    No sampling, no cache publication, no re-prefill.
    """
    import torch
    from time import perf_counter
    receiver, _ = _models(combined)
    tokenizer = tokenizers["receiver"]
    ids = list(full_ids)
    targets = list(continuation_ids)
    _token_ids(ids, "full_ids")
    _token_ids(targets, "continuation_ids")
    if tuple(ids[:len(payload.prefix_ids)]) != payload.prefix_ids or len(ids) <= len(payload.prefix_ids):
        raise ValueError("Worker input must extend the exact published prefix")
    _check_cache(payload.receiver_cache, len(payload.prefix_ids) + payload.hidden_context_tokens,
                 "published receiver cache")
    prefix_length = len(payload.prefix_ids) + payload.hidden_context_tokens
    instruction_length = len(ids) - 1
    torch.cuda.synchronize()
    started = perf_counter()
    branch, _ = _extend_private(receiver, ids[:-1], payload.receiver_cache, prefix_length, {})
    output = receiver(input_ids=torch.tensor([[ids[-1]]], device="cuda:0"),
                      attention_mask=torch.ones((1, len(ids) + payload.hidden_context_tokens),
                                                dtype=torch.long, device="cuda:0"),
                      position_ids=torch.tensor([[instruction_length + payload.hidden_context_tokens]],
                                                device="cuda:0"),
                      past_key_values=branch, use_cache=True, logits_to_keep=1)
    past, logits = output.past_key_values, output.logits[:, -1, :]
    del output, branch
    total = 0.0
    per_token = []
    with torch.no_grad():
        for step, target in enumerate(targets):
            lp = torch.log_softmax(logits.float(), dim=-1)[0, target].item()
            per_token.append(lp)
            total += lp
            if step == len(targets) - 1:
                break
            position = len(ids) + step
            output = receiver(input_ids=torch.tensor([[target]], device="cuda:0"),
                              past_key_values=past, use_cache=True,
                              attention_mask=torch.ones(
                                  (1, position + 1 + payload.hidden_context_tokens),
                                  dtype=torch.long, device="cuda:0"),
                              position_ids=torch.tensor(
                                  [[position + payload.hidden_context_tokens]], device="cuda:0"),
                              logits_to_keep=1)
            past, logits = output.past_key_values, output.logits[:, -1, :]
            del output
    torch.cuda.synchronize()
    return {"sum_logprob": total, "mean_logprob": total / max(len(targets), 1),
            "token_count": len(targets), "per_token_logprob": per_token,
            "wall_seconds": perf_counter() - started}
