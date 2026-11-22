"""Native Qwen tool envelope over the existing action contract, without action repair."""
from copy import deepcopy
import json
import re

WORKER_ROLE_SYSTEM = (
    "The current invocation is Worker. Earlier Planner function signatures and tool calls "
    "are historical background, not tools available to you in this invocation. "
    "Planner JSON, analysis and action fields are background only; do not copy them "
    "or emit a Planner action in the Worker response. "
    "Carry out the explicit Worker assignment and output contract in the following user message. "
    "The final Worker object must use the report or need_info contract, never a Planner task object. "
    "Compute the requested value yourself from the granted public state and put the exact "
    "final number in the report; do not restate the assignment or re-emit a task template "
    "as your answer. Use need_info only when a field the command requires is literally "
    "absent from the granted public state; the granted state contains every field needed. "
    "You cannot delegate, finish the parent task, or invoke Planner functions. "
    "Your output is an unapplied Worker proposal or report, not a coordinator action."
)


def native_tool_input(messages, schema, *, feedback=False, thinking=False):
    tools = []
    for variant in schema["anyOf"]:
        parameters = deepcopy(variant)
        name = parameters["properties"].pop("action")["enum"]
        if len(name) != 1:
            raise ValueError("Action variant needs one function name")
        parameters["required"].remove("action")
        tools.append({"type": "function", "function": {
            "name": name[0], "description": "Execute the " + name[0] + " action under the current runtime contract.",
            "parameters": parameters}})
    result = deepcopy(messages)
    content = json.loads(result[-1]["content"])
    old = content["instructions"]
    marker = "Return exactly one JSON action."
    native_contract = (
        "Call exactly one of the provided functions using its declared parameters. "
        "Use the native tool_call envelope. Do not emit a bare action JSON, multiple calls or prose. "
        "Function availability is an interface condition, not a recommendation to choose it.")
    if thinking:
        native_contract = (
            "Use your native internal thinking channel before the final response. "
            "The following output restrictions apply only to the final response after </think>: "
            + native_contract)
    review_marker = "Return delegate, wait, or finish JSON."
    review = review_marker in old
    if review:
        # Preserve every review instruction and the existing explicit contract.
        content["instructions"] = old.replace(review_marker, native_contract, 1)
        content["available_action_schema_scope"] = (
            "Internal runtime action representation: its action field corresponds to the native function name. "
            "For your output use the provided native function signature; do not include action inside arguments.")
        last = content["current_state"]["last_action"]
    elif marker in old:
        content["instructions"] = old.split(marker, 1)[0] + native_contract
        last = content.get("last_decision_and_result")
    else:
        raise ValueError("Native tool adapter requires a known explicit action contract")
    if feedback and last is not None:
        # These are records from the scheduler, not fabricated/model-inferred
        # results. The current observation may be newer than the action result.
        action = last["action"]
        if review:
            current = {"current_state": content.pop("current_state")}
            current["current_state"].pop("last_action")
        else:
            current = {key: content.pop(key) for key in
                       ("current_state", "budget", "planner_working_note")}
            content.pop("last_decision_and_result")
        result[-1]["content"] = json.dumps(content, ensure_ascii=False)
        result.append({"role": "assistant", "content": "", "tool_calls": [{
            "type": "function", "function": {"name": action["action"],
                "arguments": {key: value for key, value in action.items() if key != "action"}}}]})
        result.append({"role": "tool", "name": action["action"], "content": json.dumps({
            "invocation_role": content["invocation_role"],
            "recorded_execution": {"function": action["action"],
                **{key: value for key, value in last.items() if key != "action"}},
            "current_runtime_observation": current,
            "observation_scope": "Observed before this next decision; later asynchronous reports belong to this observation, not necessarily to the earlier function return.",
            "history_origin": "Recorded previous call/result re-encoded as messages, not original historical autoregressive KV."
        }, ensure_ascii=False)})
    else:
        result[-1]["content"] = json.dumps(content, ensure_ascii=False)
    return result, tools


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON key: " + key)
        value[key] = item
    return value


def parse_native_action(output, schema):
    from jsonschema import Draft202012Validator
    match = re.fullmatch(r"\s*<tool_call>\s*(\{[\s\S]*\})\s*</tool_call>\s*", output)
    if not match:
        raise ValueError("Expected exactly one complete native tool call and no surrounding prose")
    call = json.loads(match.group(1), object_pairs_hook=_unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Non-JSON constant: " + value)))
    if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
        raise ValueError("Native call needs exactly name and arguments")
    if not isinstance(call["name"], str) or not isinstance(call["arguments"], dict) or "action" in call["arguments"]:
        raise ValueError("Invalid function name or arguments")
    action = {"action": call["name"], **call["arguments"]}
    Draft202012Validator(schema).validate(action)
    return action


def separate_native_thinking(transcript, tokenizer):
    """Select the final channel by actual token boundary; never rewrite the cache."""
    marker = tokenizer.encode("</think>", add_special_tokens=False)
    opening = tokenizer.encode("<think>", add_special_tokens=False)
    if (len(marker) != 1 or len(opening) != 1 or transcript.generated_ids.count(marker[0]) != 1
            or transcript.generated_ids.count(opening[0]) != 1 or transcript.generated_ids[0] != opening[0]):
        raise ValueError("Expected a single actually generated thinking block starting the assistant turn")
    boundary = transcript.generated_ids.index(marker[0])
    if boundary < 1:
        raise ValueError("Thinking close precedes its opening")
    transcript.analysis_token_range = (len(transcript.prompt_ids)+1, len(transcript.prompt_ids) + boundary)
    transcript.action_output = tokenizer.decode(transcript.generated_ids[boundary + 1:], skip_special_tokens=True)
    return transcript.action_output
