"""Label actual cache positions by public provenance, without re-encoding KV."""
import re
from .workload import document


def planning_scope(transcript, paper, tokenizer):
    prompt = transcript.prompt_text
    if prompt is None:
        raise ValueError("Scope requires the actual initial template text; continuation suffix is not full history")
    body = document(paper)
    if prompt.count(body) != 1:
        raise ValueError("Cannot identify the exact public paper in the actual prompt")
    begin, end = prompt.index(body), prompt.index(body) + len(body)
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"] != transcript.prompt_ids:
        raise ValueError("Prompt offsets do not match the actual captured token IDs")
    selected = {i for i, (a, b) in enumerate(encoded["offset_mapping"]) if begin <= a < b <= end}
    generated = tokenizer.decode(transcript.generated_ids, skip_special_tokens=False)
    spans = [(m.start(1), m.end(1)) for m in re.finditer(r'"focus"\s*:\s*"((?:[^"\\]|\\.)*)"', generated)]
    if len(spans) != len(paper["questions"]):
        raise ValueError("Cannot identify every actual Planner focus span")
    # Generated text may have a different tokenization than a fresh encode.
    # Map the actual emitted tokens; never retokenize the generated history.
    lengths = [len(tokenizer.decode(transcript.generated_ids[:i], skip_special_tokens=False))
               for i in range(len(transcript.generated_ids) + 1)]
    focus = {len(transcript.prompt_ids) + i for i, (a, b) in enumerate(zip(lengths, lengths[1:]))
             if any(left <= a < b <= right for left, right in spans)}
    if not selected or not focus:
        raise ValueError("Semantic publication must include paper and actual Planner computation")
    return tuple(sorted(selected | focus)), {
        "policy": "paper and actual generated focus fused; all other history positions receiver-native",
        "paper_tokens": len(selected), "actual_focus_tokens": len(focus),
        "actual_focus_indices": sorted(focus), "paper_indices": sorted(selected),
        "prefix_tokens": len(transcript.prefix_ids),
        "new_task_policy": "receiver-native extension of the explicit current command",
        "learned_selection": False, "sparse_projection": False,
        "exclusions_also_include_public_questions_and_goal": True,
        "causal_control_information_is_not_removed_from_selected_states": True,
        "offset_origin": "actual initial template text, with complete encoded IDs checked against the real capture",
    }
