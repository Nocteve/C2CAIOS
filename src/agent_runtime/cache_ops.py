"""Audited cache primitives, isolated from retired workload code."""
import hashlib
from time import perf_counter
import torch
from transformers import DynamicCache

def clone_cache(cache):
    result = DynamicCache()
    for index, (key, value) in enumerate(cache):
        result.update(key.clone(), value.clone(), index)
    return result


def cache_bytes(cache):
    return sum(t.numel() * t.element_size() for pair in cache for t in pair)


def cache_digest(cache):
    digest = hashlib.sha256()
    for pair in cache:
        for tensor in pair:
            digest.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def encode_prefix(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)


def request_ids(tokenizer, messages, question):
    return tokenizer.apply_chat_template(
        [*messages, {"role": "user", "content": question}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False,
    )


@torch.inference_mode()
def prefill(model, ids, prior=None, prior_length=0, *, timings=None):
    if len(ids) <= prior_length:
        raise ValueError("Prefill must add tokens")
    torch.cuda.synchronize()
    started = perf_counter()
    past = clone_cache(prior) if prior is not None else None
    torch.cuda.synchronize()
    clone_seconds = perf_counter() - started if prior is not None else 0.0
    output = model(
        input_ids=torch.tensor([ids[prior_length:]], device="cuda:0"),
        attention_mask=torch.ones((1, len(ids)), dtype=torch.long, device="cuda:0"),
        past_key_values=past, use_cache=True, logits_to_keep=1,
    )
    torch.cuda.synchronize()
    elapsed = perf_counter() - started
    if timings is not None:
        timings.update(clone_seconds=clone_seconds, forward_seconds=elapsed-clone_seconds,
                       total_seconds=elapsed)
    return output.past_key_values, elapsed


@torch.inference_mode()
def decode(model, tokenizer, ids, prefix_ids, cache, max_new_tokens=32):
    if ids[:len(prefix_ids)] != prefix_ids or len(ids) <= len(prefix_ids):
        raise ValueError("Request does not extend the cached token prefix exactly")
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    branch = clone_cache(cache)
    torch.cuda.synchronize()
    clone_seconds = perf_counter() - started
    decode_started = perf_counter()
    output = model(
        input_ids=torch.tensor([ids[len(prefix_ids):]], device="cuda:0"),
        attention_mask=torch.ones((1, len(ids)), dtype=torch.long, device="cuda:0"),
        past_key_values=branch, use_cache=True, logits_to_keep=1,
    )
    torch.cuda.synchronize()
    prefill_seconds = perf_counter() - started
    past, logits = output.past_key_values, output.logits[:, -1, :]
    del output
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    generated, first_token_seconds = [], None
    for _ in range(max_new_tokens):
        token = logits.argmax(dim=-1)
        generated.append(token.item())
        if first_token_seconds is None:
            first_token_seconds = perf_counter() - started
        if generated[-1] in eos or len(generated) == max_new_tokens:
            break
        output = model(
            input_ids=token[:, None], past_key_values=past,
            attention_mask=torch.ones((1, len(ids) + len(generated)), dtype=torch.long, device="cuda:0"),
            use_cache=True, logits_to_keep=1,
        )
        past, logits = output.past_key_values, output.logits[:, -1, :]
        del output
    torch.cuda.synchronize()
    decode_seconds = perf_counter() - decode_started
    text = tokenizer.decode(generated, skip_special_tokens=True)
    elapsed = perf_counter() - started
    return {
        "output": text,
        "generated_token_ids": generated, "output_tokens": len(generated),
        "prompt_tokens": len(ids), "cached_prefix_tokens": len(prefix_ids),
        "clone_seconds": clone_seconds, "tail_prefill_seconds_including_clone": prefill_seconds,
        "first_token_seconds": first_token_seconds, "service_seconds": elapsed,
        "decode_seconds": decode_seconds,
        "decode_timing_includes_final_input_token": True,
        "legacy_tail_prefill_includes_clone_and_decode_seed": True,
        "allocated_before_mib": allocated_before / 2**20,
        "peak_incremental_mib": (torch.cuda.max_memory_allocated()-allocated_before) / 2**20,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "stop_reason": "eos" if generated[-1] in eos else "token_limit",
    }



def slice_cache(cache, indices):
    """Select a subset of cache positions, preserving their original order."""
    if not indices:
        raise ValueError("Slice requires at least one position")
    idx = torch.tensor(list(indices), dtype=torch.long, device="cuda:0")
    if idx.ndim != 1 or int(idx.max()) >= cache.get_seq_length():
        raise ValueError("Slice indices out of range")
    result = DynamicCache()
    for layer, (key, value) in enumerate(cache):
        result.update(key.index_select(2, idx).contiguous(),
                      value.index_select(2, idx).contiguous(), layer)
    return result


@torch.inference_mode()
def extend_past(model, ids, past):
    """Forward new tokens onto an existing past that is NOT a token prefix of ids.

    Unlike ``prefill``, the existing past occupies hidden positions that do not
    appear in ``ids`` (fused Planner KV with no visible receiver tokens), so the
    attention mask spans past plus new length.  Returns a private cache; the
    input past is not written.
    """
    if len(ids) < 1:
        raise ValueError("extend_past must add tokens")
    private = clone_cache(past)
    total = private.get_seq_length() + len(ids)
    output = model(
        input_ids=torch.tensor([ids], device="cuda:0"),
        attention_mask=torch.ones((1, total), dtype=torch.long, device="cuda:0"),
        past_key_values=private, use_cache=True, logits_to_keep=1,
    )
    torch.cuda.synchronize()
    return output.past_key_values
