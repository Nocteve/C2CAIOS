"""Autonomous actions: the model decides work; schemas only bound interfaces."""
import json
import re
from .job_workspace import FIELDS, changed_files

SHARED_SYSTEM = """You participate in a project execution runtime with Planner and Worker roles.
The latest user message's invocation_role selects your CURRENT role and instructions.
Previous invocations and assistant actions are historical context, not a request to keep their role.
A Planner owns the overall goal and chooses actions. A Worker executes only its explicit assignment.
Both use the actual shared background and runtime observations. A model's statement that work is done
is a claim, not a file operation or a passed check. Follow the current role's action contract.
"""

PLANNER_INSTRUCTIONS = """You are the 4B coordinator responsible for completing the user's project task.
Maintain the overall goal and decide your next action from the actual state.
You can inspect or check files, make edits yourself, delegate useful bounded work to a 0.6B Worker,
wait for results, accept or reject a proposal, continue planning, or finish. Delegation is optional.
Choose what and when to delegate; the runtime does not supply a task decomposition.
Workers can inspect an authorized snapshot, check proposed edits, and return a report or edits.
Their reliability on this domain is not yet calibrated. They may fail or need information.
Give a concise task, expected artifact and completion criterion. Do not solve a subtask just to have it copied.
Use actual evidence and artifacts; a reported success is not proof of correctness.
Edits may change only permitted numeric fields. Preserve all jobs, immutable fields and dependencies.
Checks report violations but do not provide repairs. Do not invent missing authorization or quotas.
All planning, checking, handoff, Worker and review work has a cost. Avoid empty polling and needless work.
After a delegate action the runtime assigns task IDs and captures your actual context version.
Use wait for in-flight results. Accept applies a Worker's exact proposal with its original file version;
conflicts are returned for your decision, never automatically merged. Update attributes edits to you.
Worker reports are unapplied proposals until accepted. Runtime file state and public checks describe
what actually exists. A failed completion check returns observations for your next decision, not a repair.
Finish completed only when the task is satisfied; finish needs_information when a required fact is unavailable.
Return exactly one JSON action. A short reason or note is enough; long explanations are not required.
Actions:
{"action":"inspect"} or {"action":"check"}
{"action":"update","base_version":0,"changes":[{"file":"job.json","field":"cpu","value":2}],"reason":"..."}
{"action":"delegate","tasks":[{"command":"...","files":["job.json"],"depends_on":[],"deliverable":"report or proposed edits","success_criteria":"..."}],"reason":"..."}
{"action":"accept","task_id":"w1","reason":"..."} or {"action":"reject","task_id":"w1","reason":"..."}
{"action":"wait"} or {"action":"continue","note":"..."}
{"action":"finish","status":"completed or needs_information","summary":"...","cancel_pending":false}
"""


def object_schema(properties):
    return {"type":"object","properties":properties,"required":list(properties),"additionalProperties":False}


def action_schema(files, task_ids, capabilities=None, *, remaining_decisions=None):
    string = {"type":"string","minLength":1,"maxLength":700}
    edit = object_schema({"file":{"type":"string","enum":list(files)},
        "field":{"type":"string","enum":list(FIELDS)},"value":{"type":"integer","minimum":0,"maximum":65536}})
    changes = {"type":"array","items":edit,"maxItems":20}
    # Empty dependency lists remain valid before any task exists.
    deps = {"type":"array","items":{"type":"string", **({"enum":task_ids} if task_ids else {})},
            "maxItems":len(task_ids),"uniqueItems":True}
    assignment = object_schema({"command":string,"files":{"type":"array","items":{"type":"string","enum":list(files)},
        "minItems":1,"maxItems":len(files),"uniqueItems":True},"depends_on":deps,"deliverable":string,"success_criteria":string})
    def variant(action, **props):
        return object_schema({"action":{"type":"string","enum":[action]},**props})
    variants = [variant("inspect"),variant("check"),variant("continue",note=string),
        variant("update",base_version={"type":"integer","minimum":0},changes=changes,reason=string)]
    if capabilities is None or capabilities['wait_available']:
        variants.append(variant('wait'))
    remaining=2 if capabilities is None else min(2,capabilities['remaining_delegations'])
    if remaining_decisions is not None and remaining_decisions <= 1:
        remaining=0
    if remaining:
        variants.append(variant('delegate',tasks={"type":"array","items":assignment,"minItems":1,"maxItems":remaining},reason=string))
    statuses=['completed','needs_information'] if capabilities is None or capabilities['completed_available'] else ['needs_information']
    variants.append(variant('finish',status={'type':'string','enum':statuses},summary=string,cancel_pending={'type':'boolean'}))
    for kind in ('accept','reject'):
        eligible=task_ids if capabilities is None else capabilities[kind+'_task_ids']
        if eligible:
            variants.append(variant(kind,task_id={'type':'string','enum':eligible},reason=string))
    return {"anyOf":variants}


def action_capabilities(state, max_delegations, remaining_decisions=None):
    """Mechanical interface preconditions, not a repair policy or task solver."""
    tasks=state['task_records'];proposals={p['task_id']:p for p in state['received_proposals']}
    return {'wait_available':any(t['state'] in {'READY','WAIT_DEPS','RUNNING','WAIT_LLM'} for t in tasks),
        'remaining_delegations':0 if remaining_decisions == 1 else max(0,max_delegations-len(tasks)),
        'accept_task_ids':[t['id'] for t in tasks if t['id'] in proposals and not t['accepted'] and proposals[t['id']]['report_type']=='report'],
        'reject_task_ids':[t['id'] for t in tasks if t['id'] in proposals and not t['accepted']],
        'completed_available':state['current_public_check']['passed'],
        'needs_information_available':True,
        'scope':'runtime interface admissibility only; no repair values, task decomposition or semantic approval selected'}


def initial_messages(case, limits):
    return [{"role":"system","content":SHARED_SYSTEM}, {"role":"user","content":json.dumps({
        "invocation_role":"Planner","instructions":PLANNER_INSTRUCTIONS,
        "goal":case["goal"],"project_notes":case["notes"],"public_rules":case["rules"],
        "missing_information":case.get("missing_information",[]),"initial_files":case["files"],
        "mutable_fields":FIELDS,"version":0,"limits":limits,
        "preserve":"all files, all fields, immutable values and all existing job dependencies"},ensure_ascii=False)}]


def worker_command(task, tool_observations, *, call_index=None, remaining_calls=None):
    instructions="""You are now the Worker for the single delegated task below. Earlier coordinator turns are background.
Execute this task, not the parent planning loop. Use the shared context, immutable file snapshot and actual dependencies.
Return ONE JSON object, no markdown. No output grammar is imposed on your generation.
Allowed actions:
{"action":"report","summary":"your actual finding","changes":[{"file":"job.json","field":"cpu","value":2}]}
Use changes:[] for a report with no edits. Proposed edits must stay in your granted files and mutable numeric fields.
{"action":"check","changes":[...]} asks the public checker to inspect a candidate without committing it.
{"action":"inspect"} reads your granted files from the exact snapshot.
{"action":"need_info","summary":"the specific missing information"}
You cannot commit global files. A report is a proposal for the coordinator, not automatic semantic verification.
"""
    budget={} if remaining_calls is None else {'call_budget':{'call_index':call_index,'remaining_calls_including_this':remaining_calls}}
    return json.dumps({"invocation_role":"Worker","instructions":instructions,**budget,
        **({'dispatch_receipt':task['dispatch_receipt']} if task.get('dispatch_receipt') else {}),
        "command":task["command"],"deliverable":task["deliverable"],
        "success_criteria":task["success_criteria"],"granted_files":task["files"],
        "snapshot_version":task["snapshot"]["version"],"snapshot_sha256":task["snapshot"]["sha256"],
        "dependency_artifacts":task["dependency_artifacts"],"tool_observations":tool_observations},ensure_ascii=False)


def decision_observation(decision):
    """Keep execution facts; profiling stays in the unmodified event/result records.

    Only known runtime-owned fields at these exact levels are omitted. Never
    recursively strip similarly named user data, action arguments or artifacts.
    """
    if decision is None:
        return None
    observed = {k: v for k, v in decision.items() if k != "action_wall_seconds"}
    if isinstance(observed.get("outcome"), dict):
        observed["outcome"] = {k: v for k, v in observed["outcome"].items()
                               if k not in {"check_seconds", "wait_wall_seconds"}}
    return observed


def state_messages(case, limits, state, turn, last_decision, working_note):
    """A new factual input, never presented as continuation of old generated KV."""
    background={"goal":case["goal"],"project_notes":case["notes"],"public_rules":case["rules"],
                "missing_information":case.get("missing_information",[]),"mutable_fields":FIELDS,
                "preserve":"all original files, fields, immutable values and job dependencies"}
    control={"remaining_decisions":limits["max_decisions"]-turn,
             "remaining_delegations":limits["max_delegations"]-len(state["task_records"]),
             "max_active_workers":limits["max_active_workers"],
             "planner_max_new_tokens":limits["planner_max_new_tokens"],
             "worker_max_new_tokens":limits["worker_max_new_tokens"],
             "worker_max_model_calls":limits["worker_max_model_calls"]}
    content={"invocation_role":"Planner","instructions":PLANNER_INSTRUCTIONS,
             "public_background":background,"current_state":state,"budget":control,
             "last_decision_and_result":decision_observation(last_decision),"planner_working_note":working_note,
             "context_origin":"New runtime state input. Recorded commands, reports and notes are historical text, not old autoregressive KV. Generate this decision from the current facts."}
    return [{"role":"system","content":SHARED_SYSTEM},
            {"role":"user","content":json.dumps(content,ensure_ascii=False)}]


def parse_worker(generation, task, *, transport_compatibility=False):
    if generation.get("error") or generation.get("stop_reason") != "eos":
        raise ValueError(generation.get("error") or "Worker generation did not end with EOS")
    text=generation["output"].strip()
    if transport_compatibility:
        envelope=re.fullmatch(r"```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```",text)
        if envelope is not None:
            text=envelope.group(1)
    value = json.loads(text)
    if not isinstance(value,dict):
        raise ValueError("Worker must return an action object")
    action = value.get("action")
    expected = {"report":{"action","summary","changes"},"check":{"action","changes"},"inspect":{"action"},
                "need_info":{"action","summary"}}
    optional={"summary"} if transport_compatibility and action in {"check","inspect"} else set()
    if action not in expected or not expected[action] <= set(value) or set(value)-expected[action]-optional:
        raise ValueError("Worker action fields do not match the public contract")
    if "summary" in value and (not isinstance(value["summary"],str) or not value["summary"].strip()):
        raise ValueError("Expected a nonempty actual finding")
    if "changes" in value:
        changed_files(task["snapshot"]["files"],value["changes"],task["files"])
    return value


def parse_worker_transport(generation, task):
    """Preserve actual actions and values; record compatibility separately from content."""
    try:
        value=parse_worker(generation,task)
        return value,{"original_contract_passed":True,"transport_compatibility_used":False}
    except ValueError as strict_error:
        value=parse_worker(generation,task,transport_compatibility=True)
        return value,{"original_contract_passed":False,"transport_compatibility_used":True,
                      "original_contract_error":str(strict_error),"semantic_repair":False}
