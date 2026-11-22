"""Optional, bounded analysis before an autonomous action. No task solver."""
from copy import deepcopy
import json

ANALYSIS_INSTRUCTIONS = (
    "Before choosing your next action, write a brief planning note in plain text. "
    "Use the actual state and observations to identify what remains unresolved. "
    "Decide whether a specific bounded part can usefully be done by the 0.6B Worker, "
    "or whether you should act yourself. If delegating, distinguish that local task "
    "from the whole goal and identify how its result will be checked and used. "
    "If reviewing an existing report, compare its claims with the supplied facts. "
    "Do not assume the Worker is correct or pre-solve its assignment just for copying. "
    "A short note is enough; do not output an action yet. No action is executed in this phase. "
    "The action_instructions field describes your available actions for the NEXT phase."
)

# 2026-10-17: sender-side arithmetic scaffold (frozen constant, append-only via
# extra_instruction, same channel as the multi guidance). Forces the Planner to
# write out every decision value with explicit steps, so a rich analysis note
# actually contains the computed value the C2C arms can carry.
ARITHMETIC_SCAFFOLD_INSTRUCTION = (
    " If your next-action decision depends on any number from the state or "
    "observations, you MUST compute that value in this note before deciding: "
    "quote each operand together with the state field it comes from, write one "
    "arithmetic step per operation showing the running result (for example "
    "'a 120 + b 90 = 210'), and finish with the final value and the branch or "
    "plan it selects. Do not decide on a value that is not written out with its "
    "arithmetic in this note."
)

FEEDBACK_REVIEW_INSTRUCTION = (
    " You are now reviewing a delivered Worker report together with the public "
    "OS gate verdict on that report, supplied in the worker_observation. First, "
    "quote or restate at least two concrete facts from the Worker report itself "
    "(task ids and their states such as encode/rank/running/blocked, or specific "
    "numbers it found). Second, the gate verdict is authoritative: if the gate "
    "rejected the report you must NOT finish the parent task; choose to "
    "re-delegate a bounded repair or verification task that names the failed "
    "check and the evidence, take over the check yourself, or inspect the state. "
    "Only choose finish when the gate accepted the report AND every dependency "
    "of the goal is terminal-success in the public state."
)

ACTION_MESSAGE = [{"role": "user", "content": (
    "Now choose and emit ONE JSON action under the action_instructions and actual state above. "
    "Your planning note is a fallible proposal, not new evidence or a completed action. "
    "Self-execution and zero delegation remain valid."
)}]


def analysis_messages(messages, extra_instruction=""):
    result = deepcopy(messages)
    state = json.loads(result[-1]["content"])
    if result[-1]["role"] != "user" or state.get("invocation_role") != "Planner":
        raise ValueError("Analysis requires the real current Planner state")
    state["action_instructions"] = state.pop("instructions")
    # ANALYSIS_INSTRUCTIONS stays a frozen constant; callers may only APPEND
    # mode-specific guidance (e.g. multi-task dispatch), never alter it.
    state["instructions"] = ANALYSIS_INSTRUCTIONS + extra_instruction
    result[-1]["content"] = json.dumps(state, ensure_ascii=False)
    return result
