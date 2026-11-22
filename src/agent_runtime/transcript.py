"""Capture actual Planner decode KV, including the final generated token."""

from dataclasses import dataclass
import json
from time import perf_counter

import torch

from .cache_ops import clone_cache, prefill
from .records import fingerprint


@dataclass
class PlannerTranscript:
    prompt_ids: list[int]
    generated_ids: list[int]
    prefix_ids: list[int]
    source_cache: object
    output: str
    metrics: dict
    prompt_text: str | None = None
    planning_input_ids: list[int] | None = None
    planning_input_text: str | None = None
    analysis_token_range: tuple[int, int] | None = None
    action_output: str | None = None


def aligned_ids(text, source_tokenizer, receiver_tokenizer):
    source = source_tokenizer.encode(text, add_special_tokens=False)
    receiver = receiver_tokenizer.encode(text, add_special_tokens=False)
    if source != receiver:
        raise ValueError("This harness requires exact shared-transcript token alignment")
    return source


@torch.inference_mode()
def capture_plan(model, source_tokenizer, receiver_tokenizer, messages, max_new_tokens, max_input_tokens,
                 producer_role="Planner", prior_prefix_ids=None, prior_cache=None, allowed_tokens=None, tools=None,
                 thinking=False, sampling=None):
    if thinking and (allowed_tokens is not None or tools is None):
        raise ValueError("Native thinking requires free decoding with native tools")
    text = source_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=thinking, **({"tools":tools} if tools is not None else {}))
    if thinking and not text.endswith("<|im_start|>assistant\n"):
        raise ValueError("Unsupported native thinking template opening")
    suffix_ids = aligned_ids(text, source_tokenizer, receiver_tokenizer)
    if (prior_cache is None) != (prior_prefix_ids is None):
        raise ValueError("Continuation requires both exact prefix IDs and their cache")
    reused_tokens = len(prior_prefix_ids) if prior_prefix_ids is not None else 0
    if prior_cache is not None and (prior_cache.get_seq_length() != reused_tokens or not text.startswith("<|im_start|>user\n")):
        raise ValueError("Unsupported or inconsistent source continuation")
    prompt_ids = (prior_prefix_ids or []) + suffix_ids
    if len(prompt_ids) + max_new_tokens + 1 > max_input_tokens:
        raise ValueError("Planner input plus generation reserve exceeds the context budget")
    torch.cuda.synchronize()
    started = perf_counter()
    kwargs = {}
    if prior_cache is not None:
        # Branch before extending: published working-note KV must stay immutable.
        kwargs = {"past_key_values": clone_cache(prior_cache),
                  "attention_mask": torch.ones((1, len(prompt_ids)), device="cuda:0", dtype=torch.long)}
    output = model(input_ids=torch.tensor([suffix_ids], device="cuda:0"), use_cache=True, logits_to_keep=1, **kwargs)
    cache, logits = output.past_key_values, output.logits[:, -1, :]
    torch.cuda.synchronize()
    prefill_seconds = perf_counter() - started
    del output
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    generated = []
    warpers = []
    if sampling is not None:
        from transformers import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
        warpers = [TemperatureLogitsWarper(sampling['temperature']),
                   TopKLogitsWarper(sampling['top_k']), TopPLogitsWarper(sampling['top_p'])]
    for _ in range(max_new_tokens):
        if allowed_tokens is not None:
            allowed = allowed_tokens(0, torch.tensor(prompt_ids + generated, device="cuda:0"))
            constrained = torch.full_like(logits, float("-inf"))
            constrained[:, allowed] = logits[:, allowed]
            logits = constrained
        if warpers:
            logits = logits.float()
            for warper in warpers:
                logits = warper(None, logits)
            token = torch.multinomial(torch.softmax(logits.float(), dim=-1), num_samples=1).squeeze(-1)
        else:
            token = logits.argmax(-1)
        generated.append(token.item())
        # Standard generation often leaves the last emitted token uncached.
        # Feed EVERY token, even EOS, before publishing the transcript handle.
        output = model(input_ids=token[:, None], past_key_values=cache, use_cache=True,
                       attention_mask=torch.ones((1, len(prompt_ids) + len(generated)), device="cuda:0", dtype=torch.long),
                       logits_to_keep=1)
        cache, logits = output.past_key_values, output.logits[:, -1, :]
        del output
        if generated[-1] in eos:
            break
    torch.cuda.synchronize()
    generation_seconds = perf_counter() - started - prefill_seconds
    im_end = source_tokenizer.convert_tokens_to_ids("<|im_end|>")
    complete = generated[-1] == im_end and generated[-1] in eos
    # Never fabricate a completed assistant turn after a truncated plan.
    tail = source_tokenizer.encode("\n", add_special_tokens=False) if complete else []
    prefix_ids = prompt_ids + generated + tail
    close_seconds = 0.0
    if tail:
        cache, close_seconds = prefill(model, prefix_ids, cache, len(prompt_ids) + len(generated))
    if cache.get_seq_length() != len(prefix_ids):
        raise RuntimeError("Published Planner cache does not cover all emitted tokens")
    for token_id in set(prefix_ids):
        if source_tokenizer.convert_ids_to_tokens(token_id) != receiver_tokenizer.convert_ids_to_tokens(token_id):
            raise ValueError("Generated transcript contains model-incompatible token IDs")
    return PlannerTranscript(prompt_ids, generated, prefix_ids, cache,
                             source_tokenizer.decode(generated, skip_special_tokens=True),
                             {"prefill_seconds": prefill_seconds, "generation_seconds": generation_seconds,
                              "close_turn_seconds": close_seconds, "complete_assistant_turn": complete,
                              "generated_tokens": len(generated), "cached_tokens": len(prefix_ids),
                              "prefill_reused_tokens": reused_tokens,
                              "native_thinking": thinking, "sampling": sampling,
                              "all_generated_tokens_cached": True, "prefix_sha256": fingerprint(prefix_ids),
                              "source_cache_origin": f"actual autoregressive {producer_role} execution, not re-encoded generated text"},
                             prompt_text=text if prior_prefix_ids is None else None)


def worker_ids(tokenizer, prefix_ids, request, max_input_tokens, *, role_system=None, native_tools=False, close_dispatch=False, thinking=False):
    # Append a new turn to exact source IDs; never decode/re-tokenize history.
    messages = [{"role": "user", "content": request}]
    tools=None
    if native_tools:
        from .worker_tools import native_worker_input
        messages,tools,_=native_worker_input(request,thinking=thinking)
    if role_system is not None:
        if not isinstance(role_system, str) or not role_system.strip():
            raise ValueError("Worker role boundary must be a nonempty system instruction")
        messages.insert(0, {"role": "system", "content": role_system})
    rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True, enable_thinking=thinking,
                                             **({'tools':tools} if tools is not None else {}))
    expected = ("<|im_start|>system\n" + role_system + "<|im_end|>\n<|im_start|>user\n"
                if role_system is not None else "<|im_start|>user\n")
    if not native_tools and not rendered.startswith(expected):
        raise ValueError("Unsupported standalone Worker role/command template")
    if native_tools and (role_system is None or not rendered.startswith('<|im_start|>system\n'+role_system)):
        raise ValueError('Worker functions require their own actual system/tool boundary')
    if thinking and (not native_tools or not rendered.endswith('<|im_start|>assistant\n')):
        raise ValueError('Native Worker thinking requires the actual open assistant template and Worker tools')
    receipt_text=''
    if close_dispatch:
        receipt=json.loads(request).get('dispatch_receipt')
        if not receipt or receipt.get('function')!='delegate' or not receipt.get('registered_tasks'):
            raise ValueError('Closing a dispatch requires an actual runtime registration receipt')
        receipt_text=tokenizer.apply_chat_template([{'role':'tool','name':'delegate',
            'content':json.dumps(receipt,ensure_ascii=False)}],tokenize=False,add_generation_prompt=False)
    ids = prefix_ids + tokenizer.encode(receipt_text+rendered, add_special_tokens=False)
    if len(ids) > max_input_tokens:
        raise ValueError("Worker input exceeds context budget; no silent truncation")
    return ids

