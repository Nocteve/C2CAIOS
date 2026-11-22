"""Small read-only code-review domain on the existing process/GPU scheduler."""
from copy import deepcopy
import json
import os
from pathlib import Path
from time import perf_counter

from .autonomy import Execution
from .delegation_contract import SHARED_SYSTEM, object_schema, decision_observation
from .records import fingerprint


def review_worker(connection, task, directory):
    try:
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
        (directory/"input.json").write_text(json.dumps(task,indent=2),encoding="utf-8")
        connection.send({"type":"started","pid":os.getpid()})
        command=json.dumps({"invocation_role":"Worker",
            **({'dispatch_receipt':task['dispatch_receipt']} if task.get('dispatch_receipt') else {}),
            "instructions":"Execute only this assignment using the shared code and documentation. Return a concise plain-text technical report with source line references, reasoning and explicit uncertainties. Do not dispatch tasks. Do not claim tests were run. Earlier Planner actions are background, not your role. No JSON output is required.",
            "command":task["command"],"deliverable":task["deliverable"],"success_criteria":task["success_criteria"],
            "assigned_items":task["files"],"dependency_artifacts":task["dependency_artifacts"]})
        connection.send({"type":"model_request","call":0,"command":command,"requested_at":perf_counter()})
        generation=connection.recv()
        (directory/"generation.json").write_text(json.dumps(generation,indent=2),encoding="utf-8")
        if generation.get("error") or generation.get("stop_reason")!="eos" or not generation.get("output","").strip():
            raise ValueError(generation.get("error") or "Incomplete actual Worker report")
        artifact={"producer":task["id"],"base_version":task["snapshot"]["version"],
            "base_sha256":task["snapshot"]["sha256"],"input_artifacts":task["dependency_artifacts"],
            "context_version":task["context_version"],"value":{"action":"report","summary":generation["output"]},
            "semantic_verified":False,"assigned_items":task["files"]}
        artifact["sha256"]=fingerprint(artifact)
        (directory/"artifact.json").write_text(json.dumps(artifact,indent=2),encoding="utf-8")
        connection.send({"type":"result","artifact":artifact,"pid":os.getpid(),"model_calls":1})
    except Exception as exc:
        connection.send({"type":"failed","error":str(exc),"pid":os.getpid()})
    finally:
        connection.close()


class ReviewInput:
    version=0
    def __init__(self,case):self.files=deepcopy(case["files"])
    def snapshot(self):return {"version":0,"files":deepcopy(self.files),"sha256":fingerprint(self.files)}


class ReviewExecution(Execution):
    def schema(self,turn):
        string={"type":"string","minLength":1,"maxLength":1200}
        def action(name,**props):return object_schema({"action":{"type":"string","enum":[name]},**props})
        ids=list(self.tasks);delivered=[t for t in ids if self.tasks[t].get("artifact")]
        # The pinned grammar accepts a first array item even with maxItems=0.
        # Use an explicit absent value; preserve it in the raw model action.
        references=({"type":"array","items":{"type":"string","enum":delivered},
                     "maxItems":len(delivered),"uniqueItems":True} if delivered else {"type":"null"})
        citation=object_schema({"source_id":{"type":"string","minLength":1},
            "first_line":{"type":"integer","minimum":1},"last_line":{"type":"integer","minimum":1}})
        finding=object_schema({"item":{"type":"string","enum":list(self.case["files"])},
            "conclusion":string,"reasoning":string,
            "citations":{"type":"array","items":citation,"minItems":1,"maxItems":6},
            "uncertainty":string})
        variants=[action("finish",findings={"type":"array","items":finding,
            "minItems":len(self.case["files"]),"maxItems":len(self.case["files"])},used_artifacts=references)]
        if self.unfinished():variants.append(action("wait"))
        remaining=self.limits["max_delegations"]-len(self.tasks)
        # Only the final decision cannot start work it has no opportunity to read.
        if remaining and turn<self.limits["max_decisions"]-1:
            assignment=object_schema({"command":string,"files":{"type":"array","items":{"type":"string","enum":list(self.case["files"])},"minItems":1,"maxItems":len(self.case["files"]),"uniqueItems":True},
                "depends_on":{"type":"array","items":{"type":"string",**({"enum":ids} if ids else {})},"maxItems":len(ids),"uniqueItems":True},
                "deliverable":string,"success_criteria":string})
            variants.append(action("delegate",tasks={"type":"array","items":assignment,"minItems":1,"maxItems":min(2,remaining)},reason=string))
        return {"anyOf":variants}

    def validate_findings(self, findings):
        """Resolve literal source ranges only; never assess or repair conclusions."""
        if not isinstance(findings,list) or len(findings)!=len(self.case['files']):
            raise ValueError('One finding per requested item is required')
        if sorted(f['item'] for f in findings)!=sorted(self.case['files']):
            raise ValueError('Findings must cover each requested item exactly once')
        resolved=[]
        for finding in findings:
            if not finding.get('citations'):
                raise ValueError('Each finding requires actual source citations')
            for citation in finding['citations']:
                first,last=citation['first_line'],citation['last_line']
                if type(first) is not int or type(last) is not int or first<1 or last<first:
                    raise ValueError('Invalid source line range')
                matches=[s for s in self.case['background']['sources']
                         if s['source_id']==citation['source_id'] and s['first_line']<=first<=last<=s['last_line']]
                if len(matches)!=1:
                    raise ValueError('Citation must be wholly inside one supplied source: '+json.dumps(citation))
                source=matches[0]
                lines=source['text'].splitlines()[first-source['first_line']:last-source['first_line']+1]
                if len(lines)!=last-first+1 or any(not line.startswith(f'L{first+i}:') for i,line in enumerate(lines)):
                    raise ValueError('Supplied source line identity is inconsistent')
                source_hash = source.get('source_sha256', source.get('sha256'))
                if not source_hash:
                    raise ValueError('Supplied source is missing its locked SHA-256')
                resolved.append({'item':finding['item'],**citation,'revision':source['revision'],
                                 'source_sha256':source_hash,'actual_lines':lines})
        return resolved

    def run_case(self,case,directory):
        self.case,self.directory=case,Path(directory);self.directory.mkdir(parents=True,exist_ok=False)
        self.workspace=ReviewInput(case);self.tasks,self.observations,self.live={},[],{}
        self.completed_counter=0;self.final_action=None;self.service_records=[]
        self.case_started=perf_counter();decisions=[];self.current_decisions=decisions
        for i in range(len(self.canceled)):self.canceled[i]=0
        status,error="budget_exhausted",None
        try:
            self.await_reply(self.submit("begin",method=self.active_method))
            for turn in range(self.limits["max_decisions"]):
                self.remaining_decisions=self.limits["max_decisions"]-turn
                if perf_counter()-self.case_started>self.limits["case_deadline_seconds"]:
                    error="parent wall budget exceeded";break
                completed_before=self.completed_counter
                state={"tasks":deepcopy(list(self.tasks.values())),"last_action":decisions[-1] if decisions else None,
                       "remaining_decisions":self.limits["max_decisions"]-turn}
                if self.limits.get('planner_interface') == 'native_tools':
                    state['last_action'] = decision_observation(state['last_action'])
                visible_artifacts={t["id"]:t["artifact"]["sha256"] for t in state["tasks"] if t.get("artifact")}
                schema=self.schema(turn)
                content={"invocation_role":"Planner",
                    "instructions":"Deliver the user's scoped code review. Decide whether bounded work is worth delegating to the 0.6B Worker, or review it yourself. You may finish without any delegation. Use actual code and reports; Worker claims can be wrong. The source is read-only. No tests or installations are available in this task. Return delegate, wait, or finish JSON. finish contains findings, exactly one per requested item, and IDs of actual Worker artifacts you used. Each finding needs item, conclusion, reasoning, citations (source_id, first_line, last_line), and uncertainty. Use exact source IDs and supplied line ranges. A citation's existence does not establish the claim; explain the code behavior supporting your conclusion. A malformed citation returns an error for your next decision, never a corrected citation. A valid finish ends the review and cancels unfinished tasks; paid work remains counted. When no artifact is visible in this input, used_artifacts must be null. To use a pending Worker's report, choose wait. All text you add or correct in finish is your work, not a Worker success. Be concise. You may delegate again within remaining budgets after reading a report; only the final decision cannot start a new Worker.",
                    "action_contract":{
                        "delegate":"Start the tasks you specify, not finish the parent review. reason explains why this work is useful. Each task needs command (an executable natural-language assignment), files (assigned item IDs), depends_on (previous task IDs whose actual reports it needs, or []), deliverable (what the Worker must return), and success_criteria (how to assess that work). Workers cannot finish the parent task. Choose a bounded assignment rather than a bare action name; its criteria should permit reporting contrary evidence.",
                        "wait":"Wait for actual pending Worker results. Does not create or complete a task.",
                        "finish":"Submit your own integrated findings for every requested item using the fields described above. This is the action for completing the parent yourself, including zero delegation. used_artifacts names only reports actually visible in current_state, or null if none are visible.",
                        "availability":"Only actions and IDs allowed by the following schema are currently callable."},
                    "available_action_schema":schema,
                    "public_task":{"goal":case["goal"],"notes":case.get("notes",""),
                                   "requested_items":case["files"],"rules":case.get("rules",[]),
                                   "source_family":case["background"].get("source_family")},
                    "public_background":case["background"],"current_state":state}
                messages=[{"role":"system","content":SHARED_SYSTEM},{"role":"user","content":json.dumps(content)}]
                reply=self.await_reply(self.submit("planner",messages=messages,schema=schema))
                if reply.get("error"):status,error="planner_failed",reply["error"];break
                action=reply["action"];self.emit("planner_action",turn=turn,action=action,published_version=reply["published_version"])
                started=perf_counter()
                try:
                    if action["action"]=="finish":
                        references=action["used_artifacts"]
                        if references is None:
                            references=[]
                        if not isinstance(references,list) or any(not isinstance(t,str) for t in references):
                            raise ValueError("used_artifacts must be null or an array of visible artifact IDs")
                        if len(references)!=len(set(references)):
                            raise ValueError("Duplicate artifact references")
                        if any(t not in visible_artifacts or self.tasks[t].get("artifact",{}).get("sha256")!=visible_artifacts[t]
                               for t in references):
                            raise ValueError("Cannot cite an artifact absent from this decision's input or with changed identity")
                        citations=self.validate_findings(action['findings'])
                        self.final_action=action
                        outcome={"reported":True,"semantic_correctness":"unassessed","producer":"Planner",
                                 "resolved_citations":citations,"citation_validation_only":True,
                                 "used_artifact_hashes":{t:visible_artifacts[t] for t in references}}
                    elif action["action"] in {"delegate","wait"}:
                        outcome=self.apply_action(action,reply["published_version"],completed_before)
                    else:raise ValueError("Unknown review action")
                    accepted=True
                except (ValueError,KeyError,TypeError) as exc:outcome={"error":str(exc)};accepted=False
                decision={"turn":turn,"action":action,"action_accepted":accepted,"outcome":outcome,
                          "input_visible_artifact_hashes":visible_artifacts,"action_wall_seconds":perf_counter()-started}
                decisions.append(decision);self.emit("action_result",**decision)
                if self.final_action:status="reported";break
        finally:
            for tid in self.unfinished():self.cancel(tid,"Review ended or exhausted budget")
        while self.pending:self.pump()
        audit=self.await_reply(self.submit("end"))
        result={"id":case["id"],"method":self.active_method,"status":status,"error":error,
            "final_action":self.final_action,"decisions":decisions,"tasks":list(self.tasks.values()),
            "delegations":len(self.tasks),"model_service":self.service_records,"context_audit":audit,
            "c2c_executed":any(v.get("fused") for v in audit.get("versions",[])),
            "completed_goal":None,"declared_information_block_with_known_missing":False,
            "semantic_correctness":"unassessed","observed_parent_wall_seconds":perf_counter()-self.case_started,
            "scope":"one real scoped review, independent autonomous trajectory; no host semantic success oracle"}
        (self.directory/"result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
        if self.final_action:
            sections=[]
            for finding in self.final_action['findings']:
                locations='; '.join(f"{c['source_id']}:{c['first_line']}-{c['last_line']}" for c in finding['citations'])
                sections.append(f"## {finding['item']}\n\n{finding['conclusion']}\n\n{finding['reasoning']}\n\nSources: {locations}\n\nUncertainty: {finding['uncertainty']}")
            (self.directory/'report.md').write_text('\n\n'.join(sections)+'\n',encoding='utf-8')
        return result
