"""Model-directed delegation with real processes, events and versioned artifacts."""
from copy import deepcopy
import hashlib
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
from time import perf_counter

from .project import ROOT
from .records import fingerprint
from .job_workspace import (JobWorkspace, changed_files, check_files,
                            proposal_effect, proposal_effect_observation, proposal_check_observation,
                            proposal_freshness_observation)
from .delegation_contract import initial_messages, state_messages, action_schema, action_capabilities, worker_command, parse_worker_transport
from .model_service import gpu_service


def delegated_worker(connection, task, case, limits, directory):
    if os.name=="posix":
        import resource
        resource.setrlimit(resource.RLIMIT_AS,(256*1024**2,256*1024**2))
        resource.setrlimit(resource.RLIMIT_CPU,(30,30))
    if limits.get("domain") == "code_review":
        from .review_execution import review_worker
        return review_worker(connection, task, directory)
    if limits.get('domain') == 'ledger_component':
        from .ledger_worker import ledger_worker
        return ledger_worker(connection,task,case,limits,directory)
    observations=[]
    try:
        directory=Path(directory)
        directory.mkdir(parents=True,exist_ok=False)
        (directory/"input.json").write_text(json.dumps(task,ensure_ascii=False,indent=2),encoding="utf-8")
        connection.send({"type":"started","pid":os.getpid()})
        for call in range(limits["worker_max_model_calls"]):
            remaining=limits['worker_max_model_calls']-call
            budget_args={'call_index':call,'remaining_calls':remaining} if limits.get('worker_interface')=='native_tools' else {}
            connection.send({"type":"model_request","call":call,"command":worker_command(task,observations,**budget_args),
                             "requested_at":perf_counter()})
            response=connection.recv()
            if response.get("error"):
                raise ValueError(response["error"])
            if limits.get('worker_interface')=='native_tools':
                from .worker_tools import parse_native_worker
                value,output_validation=parse_native_worker(response,task,remaining_calls=remaining)
            else:
                value,output_validation=parse_worker_transport(response,task)
            if value["action"] in {"report","need_info"}:
                artifact={"producer":task["id"],"base_version":task["snapshot"]["version"],
                    "base_sha256":task["snapshot"]["sha256"],"input_artifacts":task["dependency_artifacts"],
                    "context_version":task["context_version"],"value":value,"semantic_verified":False,
                    "output_validation":output_validation}
                artifact["sha256"]=fingerprint(artifact)
                (directory/"artifact.json").write_text(json.dumps(artifact,ensure_ascii=False,indent=2),encoding="utf-8")
                connection.send({"type":"result","artifact":artifact,"pid":os.getpid(),"model_calls":call+1})
                return
            start=perf_counter()
            if value["action"]=="inspect":
                result={"version":task["snapshot"]["version"],
                        "files":{n:task["snapshot"]["files"][n] for n in task["files"]}}
            else:
                candidate=changed_files(task["snapshot"]["files"],value["changes"],task["files"])
                (directory/f"candidate-{call}.json").write_text(json.dumps(candidate,indent=2),encoding="utf-8")
                effect=proposal_effect(case,task["snapshot"]["files"],candidate)
                result={**proposal_check_observation(effect),"proposal_effect":proposal_effect_observation(effect)}
                result["effect_snapshot"]={"version":task["snapshot"]["version"],
                    "sha256":task["snapshot"]["sha256"],
                    "current_workspace_freshness":"not available inside Worker; coordinator revalidates"}
            observation={"worker_action":value,"tool_result":result}
            observations.append(observation)
            connection.send({"type":"tool_result",**observation,"seconds":perf_counter()-start,
                             "output_validation":output_validation})
        raise ValueError("Worker model-call budget exhausted without a terminal artifact")
    except Exception as exc:
        try:
            connection.send({"type":"failed","error":f"{type(exc).__name__}: {exc}","pid":os.getpid()})
        except (BrokenPipeError,EOFError,OSError):
            pass
    finally:
        connection.close()


class ServiceFailure(RuntimeError):
    pass


class Execution:
    def __init__(self, context, connection, service, canceled, limits, emit):
        self.ctx,self.connection,self.service,self.canceled=context,connection,service,canceled
        self.limits,self.emit=limits,emit
        self.sequence=0
        self.pending,self.replies,self.live={}, {}, {}
        self.tasks,self.observations={},[]
        self.completed_counter=0
        self.service_records=[]
        self.remaining_decisions=limits['max_decisions']

    def submit(self,kind,*,target=None,**payload):
        self.sequence+=1
        rid=f"req{self.sequence}"
        request={"id":rid,"kind":kind,"submitted_at":perf_counter(),**payload}
        self.pending[rid]={"kind":kind,"target":target,"submitted_at":request["submitted_at"]}
        self.connection.send(request)
        self.emit("model_request",request=request,target=target)
        return rid

    def reap(self,tid):
        item=self.live.pop(tid,None)
        if item is None:
            return
        process,pipe,started=item
        process.join(timeout=1)
        if process.is_alive():
            process.terminate(); process.join(timeout=2)
        if process.is_alive():
            process.kill(); process.join(timeout=2)
        pipe.close()
        task=self.tasks[tid]
        task.update(pid=process.pid,exitcode=process.exitcode,process_lifetime_seconds=perf_counter()-started)
        self.emit("process_reaped",task_id=tid,pid=process.pid,exitcode=process.exitcode,alive=process.is_alive(),
                  process_lifetime_seconds=task["process_lifetime_seconds"])
        if process.is_alive():
            raise ServiceFailure("Worker could not be reaped")
        process.close()

    def cancel(self,tid,reason):
        task=self.tasks[tid]
        self.canceled[task["slot"]]=1
        if tid in self.live:
            self.live[tid][0].terminate()
            self.reap(tid)
        task.update(state="CANCELED",error=reason)
        self.completed_counter+=1
        self.emit("task_canceled",task_id=tid,reason=reason,
                  cancellation="queued model calls skipped; an already executing model call finishes and is discarded")

    def failed(self,tid,error):
        self.tasks[tid].update(state="FAILED",error=error)
        self.canceled[self.tasks[tid]["slot"]]=1
        self.completed_counter+=1
        self.observations.append({"task_id":tid,"status":"failed","error":error})
        self.emit("task_failed",task_id=tid,error=error)
        self.reap(tid)

    def launch_ready(self):
        for tid,task in self.tasks.items():
            if task["state"] not in {"READY","WAIT_DEPS"}:
                continue
            deps=[self.tasks[d] for d in task["depends_on"]]
            if any(d["state"] in {"FAILED","NEEDS_INFO","CANCELED","BLOCKED"} for d in deps):
                task.update(state="BLOCKED",error="dependency did not deliver a report")
                self.completed_counter+=1
                self.observations.append({"task_id":tid,"status":"blocked","depends_on":task["depends_on"]})
                self.emit("task_blocked",task_id=tid,dependencies=task["depends_on"])
                continue
            if not all(d["state"]=="DONE" for d in deps):
                task["state"]="WAIT_DEPS"
                continue
            if len(self.live)>=self.limits["max_active_workers"]:
                continue
            task["dependency_artifacts"]=[d["artifact"] for d in deps]
            parent,child=self.ctx.Pipe()
            process=self.ctx.Process(target=delegated_worker,args=(child,deepcopy(task),self.case,self.limits,
                                       str(self.directory/"workers"/tid)),name="Worker-"+tid)
            started=perf_counter()
            process.start(); child.close()
            self.live[tid]=(process,parent,started)
            task.update(state="RUNNING",pid=process.pid)
            self.emit("task_started",task_id=tid,pid=process.pid,context_version=task["context_version"],
                      snapshot_version=task["snapshot"]["version"],snapshot_sha256=task["snapshot"]["sha256"],
                      input_artifacts=[a["sha256"] for a in task["dependency_artifacts"]])

    def pump(self):
        if not self.service.is_alive() and not self.connection.poll():
            raise ServiceFailure("GPU service process exited")
        connections=[self.connection]+[p for _,p,_ in self.live.values()]
        for pipe in wait(connections,timeout=.1):
            if pipe is self.connection:
                try:
                    response=pipe.recv()
                except EOFError as exc:
                    raise ServiceFailure("GPU service IPC closed") from exc
                if response.get("type")!="model_response":
                    raise ServiceFailure(str(response))
                rid=response["request_id"]
                origin=self.pending.pop(rid)
                self.emit("model_response",response=response,target=origin["target"])
                self.service_records.append({"kind":origin["kind"],"target":origin["target"],
                    "queue_seconds":response["queue_seconds"],
                    "request_round_trip_seconds":perf_counter()-origin["submitted_at"],
                    "observed_service_wall_seconds":response["observed_service_wall_seconds"],
                    "peak_allocated_mib":response["peak_allocated_mib"],
                    "output_tokens":response.get("generated_tokens_this_request",
                        ((response.get("result") or {}).get("generation") or {}).get("output_tokens",0)),
                    "error":response.get("error"),"canceled":response.get("canceled",False)})
                generation=(response.get('result') or {}).get('generation') or {}
                preparation=(response.get('result') or {}).get('context_preparation') or response.get('failed_context_preparation') or {}
                handoff=preparation.get('handoff',{}).get('metrics',{})
                self.service_records[-1].update(
                    worker_consumed_fused_cache=bool(origin['kind']=='worker' and generation.get('published_cache_fused') and generation.get('output_tokens',0)>0),
                    planner_handoff_tokens=handoff.get('generated_tokens',0),
                    planner_handoff_model_seconds=sum(handoff.get(k,0) for k in ('prefill_seconds','generation_seconds','close_turn_seconds')),
                    worker_generated_tokens=generation.get('output_tokens',0) if origin['kind']=='worker' else 0,
                    worker_decode_service_seconds=generation.get('service_seconds',0) if origin['kind']=='worker' else 0)
                if response.get("fatal"):
                    raise ServiceFailure(str(response["error"]))
                tid=origin["target"]
                if origin["kind"]=="worker":
                    if tid in self.live:
                        generated=(response.get("result") or {}).get("generation")
                        reply=generated if generated is not None else {"error":str(response.get("error") or "model request canceled")}
                        try:
                            self.live[tid][1].send(reply)
                            self.tasks[tid]["state"]="RUNNING"
                        except (BrokenPipeError,EOFError,OSError):
                            self.failed(tid,"Worker exited while receiving the model result")
                    else:
                        self.emit("model_result_discarded",request_id=rid,task_id=tid,reason="task already canceled or failed")
                else:
                    self.replies[rid]=response
            else:
                tid=next((t for t,(_,p,_) in self.live.items() if p is pipe),None)
                if tid is None:
                    continue
                try:
                    message=pipe.recv()
                except (EOFError,OSError):
                    self.failed(tid,"Worker IPC closed without a terminal artifact")
                    continue
                task=self.tasks[tid]
                if message["type"]=="model_request":
                    task["state"]="WAIT_LLM"
                    self.submit("worker",target=tid,worker_slot=task["slot"],version=task["context_version"],
                                command=message["command"],worker_call_index=message['call'],worker_requested_at=message["requested_at"])
                elif message["type"]=="tool_result":
                    self.emit("worker_tool",task_id=tid,**message)
                elif message["type"]=="result":
                    artifact=message["artifact"]
                    recorded=json.loads((self.directory/"workers"/tid/"artifact.json").read_text())
                    if recorded!=artifact or fingerprint({k:v for k,v in artifact.items() if k!="sha256"})!=artifact["sha256"]:
                        raise ServiceFailure("Worker artifact integrity error")
                    task.update(artifact=artifact,state="DONE" if artifact["value"]["action"]=="report" else "NEEDS_INFO",
                                model_calls=message["model_calls"])
                    checked_at=perf_counter()
                    evidence={"application_state":"proposal_not_applied","workspace_version":self.workspace.version}
                    if self.limits.get("domain") == "code_review":
                        evidence.update(application_state="unverified_review_report",
                                        semantic_correctness="unassessed")
                    elif artifact["value"]["action"]=="report":
                        candidate=changed_files(task["snapshot"]["files"],artifact["value"]["changes"],task["files"])
                        effect=proposal_effect(self.case,task["snapshot"]["files"],candidate)
                        evidence.update(candidate_parent_check=proposal_check_observation(effect),
                            candidate_changed_files={n:v for n,v in candidate.items() if v!=task["snapshot"]["files"][n]},
                            scope="public parent constraints on dispatch-time snapshot plus exact proposed edits; not assigned-subtask verification")
                        evidence.update(proposal_effect=proposal_effect_observation(effect),
                            proposal_snapshot_status=proposal_freshness_observation(task["snapshot"],self.workspace.snapshot(),artifact))
                    evidence["check_seconds"]=perf_counter()-checked_at
                    self.completed_counter+=1
                    self.observations.append({"task_id":tid,"status":task["state"],"artifact":artifact,"runtime_evidence":evidence})
                    self.emit("artifact_created",task_id=tid,artifact=artifact,runtime_evidence=evidence)
                    self.reap(tid)
                elif message["type"]=="failed":
                    self.failed(tid,message["error"])
        for tid,(process,_,started) in list(self.live.items()):
            if perf_counter()-started>self.limits["worker_deadline_seconds"]:
                process.terminate()
                self.failed(tid,"Worker deadline exceeded")
        if any(perf_counter()-r["submitted_at"]>self.limits["model_service_deadline_seconds"] for r in self.pending.values()):
            raise ServiceFailure("Model request deadline exceeded")
        self.launch_ready()

    def await_reply(self,rid):
        while rid not in self.replies:
            self.pump()
        response=self.replies.pop(rid)
        if response.get("error"):
            raise ValueError(str(response["error"]))
        return response["result"]

    def unfinished(self):
        return [t for t in self.tasks if self.tasks[t]["state"] in {"READY","WAIT_DEPS","RUNNING","WAIT_LLM"}]

    def observed_files(self):
        started=perf_counter()
        observation={"actual_workspace":self.workspace.snapshot(),"public_check":self.workspace.check()}
        observation["check_seconds"]=perf_counter()-started
        return observation

    def current_planner_state(self, *, include_ledger=False, remaining_decisions=None):
        """Observe actual files and exact proposals; never choose or repair an action."""
        started=perf_counter()
        snapshot=self.workspace.snapshot()
        proposals=[]
        for tid,task in self.tasks.items():
            artifact=task.get("artifact")
            if artifact is None:
                continue
            item={"task_id":tid,"artifact_sha256":artifact["sha256"],
                  "base_version":artifact["base_version"],"accepted_by_planner":task["accepted"],
                  "report_type":artifact["value"]["action"],
                  "proposal_snapshot_status":proposal_freshness_observation(task["snapshot"],snapshot,artifact)}
            if include_ledger:
                item.update(original_worker_value=artifact["value"],
                            original_base_sha256=artifact["base_sha256"],
                            review_status="accepted" if task["accepted"] else "rejected" if "rejection_reason" in task else "pending")
            if artifact["value"]["action"]=="report":
                original=task["snapshot"]["files"]
                candidate=changed_files(original,artifact["value"]["changes"],task["files"])
                effect=proposal_effect(self.case,original,candidate)
                if not include_ledger:item["proposed_edits"]=artifact["value"]["changes"]
                item.update(effective_proposed_edits=[{"file":name,"field":field,"before":original[name][field],"after":value}
                        for name,fields in candidate.items() for field,value in fields.items() if original[name][field]!=value],
                    candidate_parent_check=proposal_check_observation(effect),
                    proposal_effect=proposal_effect_observation(effect))
            proposals.append(item)
        state={"origin":"runtime observation of actual versioned files and actual received artifacts",
            "workspace":{"version":snapshot["version"],"sha256":snapshot["sha256"],"files":snapshot["files"]},
            "current_public_check":self.workspace.check(),"received_proposals":proposals,
            "check_scope":"public parent constraints; candidate checks are on their original dispatch snapshots, not subtask correctness judgments"}
        if include_ledger:
            state["task_records"]=[{k:t[k] for k in ("id","state","command","files","depends_on",
                "deliverable","success_criteria","accepted","context_version","planner_rationale",
                "error","rejection_reason","model_calls") if k in t} |
                {"snapshot_version":t["snapshot"]["version"],"snapshot_sha256":t["snapshot"]["sha256"],
                 "dependency_artifacts":t["dependency_artifacts"]} for t in self.tasks.values()]
            if self.limits.get('enforce_action_preconditions',False):
                state['action_preconditions']=action_capabilities(state,self.limits['max_delegations'],remaining_decisions)
        seconds=perf_counter()-started
        self.state_observation_seconds+=seconds
        self.emit("planner_state_observed",state=state,seconds=seconds)
        return state

    def apply_action(self,action,published,completed_before):
        kind=action["action"]
        if kind=="inspect":
            return self.workspace.snapshot()
        if kind=="check":
            return self.workspace.check()
        if kind=="continue":
            self.working_note={"text":action["note"],"producer":"Planner","workspace_version":self.workspace.version,
                               "workspace_sha256":self.workspace.snapshot()["sha256"],
                               "origin":"exact recorded continue action text, re-encoded as a model note; not authoritative runtime evidence"}
            return {"noted":action["note"]}
        if kind=="update":
            # The exact hash is from the observation at this version, not a guessed model string.
            known=self.workspace.snapshot()
            result=self.workspace.commit(action["base_version"],known["sha256"],action["changes"],"Planner")
            return {**result,**self.observed_files()}
        if kind=="delegate":
            if self.remaining_decisions <= 1:
                raise ValueError("No later Planner decision remains to receive or use a new Worker's result")
            assignments=action["tasks"]
            if len(self.tasks)+len(assignments)>self.limits["max_delegations"]:
                raise ValueError("Delegation budget exceeded")
            if not published:
                raise ServiceFailure("Delegate action has no actual Planner KV version")
            known=set(self.tasks)
            for item in assignments:
                if not item["files"] or set(item["files"])-set(self.case["files"]) or set(item["depends_on"])-known:
                    raise ValueError("Unknown file or dependency; runtime will not invent it")
            created=[]
            for item in assignments:
                tid="w"+str(len(self.tasks)+1)
                task={**item,"id":tid,"slot":len(self.tasks)+1,"state":"READY","accepted":False,
                      "snapshot":self.workspace.snapshot(),"context_version":published["id"],"dependency_artifacts":[],
                      "planner_rationale":action["reason"]}
                self.tasks[tid]=task
                created.append({"task_id":tid,"context_version":published["id"],"depends_on":item["depends_on"]})
                self.emit("task_declared",task=task,producer="Planner")
            receipt={"function":"delegate","registered_tasks":created,
                     "scope":"Runtime task registration only; no Worker result or task success yet"}
            for item in created:
                self.tasks[item['task_id']]['dispatch_receipt']=deepcopy(receipt)
            self.emit('dispatch_registered',receipt=receipt)
            self.launch_ready()
            return {"delegated":created}
        if kind=="wait":
            if self.completed_counter>completed_before:
                return {"event":"result arrived during your decision"}
            if not self.unfinished():
                raise ValueError("No unfinished tasks to wait for")
            start=perf_counter()
            while self.completed_counter==completed_before:
                self.pump()
                if perf_counter()-self.case_started>self.limits["case_deadline_seconds"]:
                    raise ValueError("Parent wall budget exceeded while waiting")
            return {"event":"task state changed","wait_wall_seconds":perf_counter()-start}
        if kind in {"accept","reject"}:
            task=self.tasks[action["task_id"]]
            if "artifact" not in task:
                raise ValueError("Task has not delivered an artifact")
            if kind=="reject":
                if task["accepted"]:
                    raise ValueError("This artifact was already accepted; use an explicit update to revise committed files. Rejection does not undo writes.")
                task.update(accepted=False,rejection_reason=action["reason"])
                return {"rejected":task["id"]}
            if task["accepted"]:
                return {"already_accepted":task["id"],"committed":False,"files_unchanged":True}
            value=task["artifact"]["value"]
            if value["action"]!="report":
                raise ValueError("A needs-information response is not a completed work report")
            if value["changes"]:
                artifact=task["artifact"]
                result=self.workspace.commit(artifact["base_version"],artifact["base_sha256"],
                    value["changes"],task["id"],task["files"])
                task["accepted"]=bool(result["committed"] or result.get("unchanged"))
            else:
                result={"accepted_report":task["id"],"semantic_verified":False}
                task["accepted"]=True
            result.update(self.observed_files())
            self.emit("planner_artifact_decision",task_id=task["id"],accepted=task["accepted"],reason=action["reason"],result=result)
            return result
        if kind=="finish":
            if action["status"]=="completed":
                observation=self.observed_files()
                if not observation["public_check"]["passed"]:
                    return {"finished":False,"error":"public_completion_check_failed",**observation,
                            "content_repaired":False,"pending_tasks_canceled":False}
            if self.unfinished() and not action["cancel_pending"]:
                raise ValueError("Choose wait or explicitly cancel unfinished tasks before finishing")
            for tid in self.unfinished():
                self.cancel(tid,"Planner finished and explicitly canceled remaining work")
            self.final_action=action
            return {"finished":True}
        raise ValueError("Unknown Planner action")

    def run_case(self,case,directory):
        self.case,self.directory=case,Path(directory)
        self.workspace=JobWorkspace(case,self.directory/"workspace")
        self.tasks,self.observations,self.live={},[],{}
        self.completed_counter,self.final_action=0,None
        self.service_records=[]
        self.state_observation_seconds=0.0
        self.working_note=None
        for i in range(len(self.canceled)):
            self.canceled[i]=0
        self.case_started=perf_counter()
        decisions=[]
        self.current_decisions=decisions
        status,error="budget_exhausted",None
        try:
            self.await_reply(self.submit("begin",method=self.active_method))
            for turn in range(self.limits["max_decisions"]):
                self.remaining_decisions=self.limits["max_decisions"]-turn
                if perf_counter()-self.case_started>self.limits["case_deadline_seconds"]:
                    error="parent wall budget exceeded"; break
                observation={"invocation_role":"Planner","new_observations":self.observations,"file_version":self.workspace.version,
                    "remaining_decisions":self.limits["max_decisions"]-turn,
                    "remaining_delegations":0 if self.remaining_decisions == 1 else self.limits["max_delegations"]-len(self.tasks),
                    "tasks":[{"id":t["id"],"state":t["state"],"depends_on":t["depends_on"],
                              "accepted":t["accepted"]} for t in self.tasks.values()]}
                self.observations=[]
                completed_before=self.completed_counter
                fresh_state=self.limits.get("planner_context_mode")=="fresh_current_state"
                if fresh_state:
                    state=self.current_planner_state(include_ledger=True,remaining_decisions=self.remaining_decisions)
                    messages=state_messages(case,self.limits,state,turn,decisions[-1] if decisions else None,self.working_note)
                else:
                    messages=initial_messages(case,self.limits) if turn==0 else [{"role":"user","content":json.dumps(observation,ensure_ascii=False)}]
                if not fresh_state and self.limits.get("planner_state_packet",False):
                    content=json.loads(messages[-1]["content"])
                    content["current_state"]=self.current_planner_state()
                    messages[-1]["content"]=json.dumps(content,ensure_ascii=False)
                capabilities=state.get('action_preconditions') if fresh_state else None
                reply=self.await_reply(self.submit("planner",messages=messages,schema=action_schema(case["files"],list(self.tasks),capabilities,
                    remaining_decisions=self.remaining_decisions)))
                if reply.get("error"):
                    status,error="planner_failed",reply["error"]; break
                action=reply["action"]
                self.emit("planner_action",turn=turn,action=action,published_version=reply["published_version"])
                started=perf_counter()
                try:
                    outcome=self.apply_action(action,reply["published_version"],completed_before)
                    accepted=True
                except (ValueError,KeyError,TypeError) as exc:
                    outcome={"error":str(exc),"content_repaired":False}; accepted=False
                elapsed=perf_counter()-started
                decision={"turn":turn,"action":action,"action_accepted":accepted,"outcome":outcome,
                          "action_wall_seconds":elapsed}
                decisions.append(decision)
                self.observations.append({"action_result":outcome})
                self.emit("action_result",**decision)
                if self.final_action:
                    status="finished"; break
        except ValueError as exc:
            status,error="model_budget_or_protocol_failed",str(exc)
        finally:
            for tid in self.unfinished():
                self.cancel(tid,"Parent finished or reached its budget")
        # Drain paid in-flight inference before switching cases; do not reuse its outputs.
        while self.pending:
            self.pump()
        audit=self.await_reply(self.submit("end"))
        check_started=perf_counter()
        check=self.workspace.check()
        check_seconds=perf_counter()-check_started
        declared=(self.final_action or {}).get("status")
        result={"id":case["id"],"status":status,"error":error,"final_action":self.final_action,
            "communication_method":self.active_method,
            "c2c_cache_prepared":any(v.get('fused',False) for v in audit['versions']),
            "c2c_executed":any(s.get('worker_consumed_fused_cache',False) for s in self.service_records),
            "decisions":decisions,"tasks":list(self.tasks.values()),"final_files":self.workspace.snapshot(),
            "terminal_check":check,"terminal_check_seconds":check_seconds,
            "completed_goal":declared=="completed" and check["passed"],
            "declared_information_block_with_known_missing":declared=="needs_information" and bool(check["missing_information"]),
            "information_reason_verified":False,
            "worker_reports":sum(t["state"]=="DONE" for t in self.tasks.values()),
            "accepted_worker_artifacts":sum(t["accepted"] for t in self.tasks.values()),
            "planner_direct_commits":sum(d["action"]["action"]=="update" and d["outcome"].get("committed",False) for d in decisions),
            "worker_proposal_commits":sum(d["action"]["action"]=="accept" and d["outcome"].get("committed",False) for d in decisions),
            "delegations":len(self.tasks),"model_service":self.service_records,"context_audit":audit,
            "state_observation_seconds":self.state_observation_seconds,
            "observed_parent_wall_seconds":perf_counter()-self.case_started,
            "scope":"autonomous engineering development; terminal checks do not establish optimal delegation"}
        (self.directory/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
        return result


def run_autonomy(output_dir, *, review=False, planner_analysis=False, review_method=None, planner_tools=False, planner_tool_feedback=False, review_data_dir=None, planner_thinking=False, worker_tools=False, communication_method=None, worker_context_mode=None, protocol_path=None):
    config_path=ROOT/("configs/scipy-review.json" if review else "configs/autonomy.json") if protocol_path is None else Path(protocol_path).resolve()
    if not config_path.is_relative_to(ROOT.resolve()) or (protocol_path is not None and review):
        raise ValueError('Configuration protocol must be inside this project and outside review mode')
    limits=json.loads(config_path.read_text())
    if review_data_dir is not None:
        if not review:
            raise ValueError('An explicit review packet requires review mode')
        directory=Path(review_data_dir).resolve()
        if not directory.is_relative_to(ROOT.resolve()):
            raise ValueError('Review public packet must be inside this project')
        limits['data_directory']=directory.relative_to(ROOT.resolve()).as_posix()
    if planner_tool_feedback and not planner_tools:
        raise ValueError('Native tool feedback requires native Planner tools')
    if worker_tools:
        if not planner_tools or review:
            raise ValueError('Worker tools require native Planner tools in the configuration application')
        limits['worker_interface']='native_tools'
        limits['worker_decode']='free native tool call, strict original scope checks, actual tool feedback; no output masking'
    if planner_tools:
        if planner_analysis:
            raise ValueError('Native tools cannot be combined with the separate analysis intervention')
        limits['planner_interface']='native_tools'
        limits['planner_decode']='free greedy native tool envelope; strict post-generation JSON Schema validation'
        if planner_tool_feedback:
            limits['planner_feedback_mode']='native_tool_response'
    if planner_thinking:
        if not planner_tools or planner_analysis:
            raise ValueError('Native thinking requires native tools without the old analysis intervention')
        policy=json.loads((ROOT/'configs/planner-thinking.json').read_text())
        limits['planner_native_thinking']=True
        limits['planner_sampling']=policy['sampling']
        limits['planner_max_new_tokens']=policy['max_new_tokens']
        limits['planner_decode']='native thinking and tool call, sampled freely; exact token-boundary parsing and strict action validation'
        limits['planner_thinking_policy']=policy
    if review_method is not None:
        if not review or review_method not in limits['methods']:
            raise ValueError('Review method must be one of the configured review arms')
        limits['methods']=[review_method]
    if planner_analysis:
        analysis_policy=json.loads((ROOT/'configs/planner-analysis.json').read_text())
        limits['planner_analysis_max_new_tokens']=analysis_policy['max_new_tokens']
        limits['planner_analysis_policy']=analysis_policy
    if communication_method is not None:
        if review or communication_method not in {'text_full','text_brief','c2c_shared'}:
            raise ValueError('Unknown configuration communication method')
        if communication_method == 'c2c_shared' and not planner_tools:
            raise ValueError('c2c_shared requires native Planner tools so the Worker receives an explicit role boundary')
        limits.update(methods=[communication_method],c2c_enabled=communication_method=='c2c_shared',handoff_max_new_tokens=768,
            scope='Autonomous configuration task with explicit selected background communication; actual execution is recorded separately')
    worker_context_mode = worker_context_mode or limits.get('worker_context_mode')
    if worker_context_mode is not None:
        if review or worker_context_mode not in {'planner_transcript', 'independent_session', 'semantic_session'}:
            raise ValueError('Unknown configuration Worker context mode')
        if worker_context_mode in {'independent_session', 'semantic_session'} and (not worker_tools or 'c2c_shared' in limits['methods']):
            raise ValueError('Independent sessions require native Worker tools and text communication')
        limits['worker_context_mode'] = worker_context_mode
        if worker_context_mode == 'semantic_session' and (limits['methods'] != ['text_full'] or not planner_tools or planner_analysis):
            raise ValueError('Semantic session currently requires fresh native configuration text_full without separate analysis')
    data=ROOT/limits["data_directory"]
    manifest=json.loads((data/"manifest.json").read_text())
    public=data/"public.json"
    if hashlib.sha256(public.read_bytes()).hexdigest()!=manifest["public_sha256"]:
        raise ValueError("Frozen task inputs changed")
    cases=json.loads(public.read_text())
    if [c["id"] for c in cases]!=manifest["expected_case_ids"]:
        raise ValueError("Frozen task denominator changed")
    output=Path(output_dir)
    output.mkdir(parents=True,exist_ok=False)
    config=json.loads((ROOT/"configs/cloud.json").read_text())
    if review:
        config["max_input_tokens"]=limits["max_input_tokens"]
    lock=json.loads((ROOT/"configs/cloud.models.lock.json").read_text())
    ctx=mp.get_context("spawn")
    parent,child=ctx.Pipe()
    canceled=ctx.Array("b",limits["max_delegations"]+1)
    process=ctx.Process(target=gpu_service,args=(child,config,lock,limits,canceled),name="GPU-owner")
    started=perf_counter()
    schedule=[(method,case) for method in limits["methods"] for case in cases]
    result={"kind":"autonomous_code_review" if review else "autonomous_delegation","status":"running","methods":limits["methods"],"protocol":limits,
        "manifest":manifest,"cases":[{"id":c["id"],"method":m,"status":"not_run"} for m,c in schedule],"training":False,
        "c2c_executed":False,"forced_delegation":False,"scheduler_pid":os.getpid()}
    identity={}
    logging_seconds=0.0
    log=(output/"events.jsonl").open("x",encoding="utf-8")
    def emit(kind,**fields):
        nonlocal logging_seconds
        before=perf_counter()
        if "type" in fields:
            fields["message_type"]=fields.pop("type")
        log.write(json.dumps({"type":kind,**identity,**fields},ensure_ascii=False,allow_nan=False)+"\n")
        log.flush(); logging_seconds+=perf_counter()-before
    def save():
        temp=output/"progress.tmp"
        temp.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
        temp.replace(output/"progress.json")
    sources=[*sorted((ROOT/"src/agent_runtime").glob("*.py")),ROOT/"scripts/run.py",ROOT/"scripts/prepare_job_tasks.py",
             config_path]
    if planner_analysis:sources.append(ROOT/'configs/planner-analysis.json')
    if planner_thinking:sources.append(ROOT/'configs/planner-thinking.json')
    emit("metadata",config=config,models=lock,limits=limits,manifest=manifest,scheduler_pid=os.getpid(),
         source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
         reference_answers_read=False,worker_output_constraints=False,separate_gpu_owner=True,
         physical_zero_copy=False,cgroup_enabled=False)
    runner=None
    try:
        process.start(); child.close()
        if not parent.poll(limits["model_service_deadline_seconds"]):
            raise ServiceFailure("GPU owner did not start in time")
        ready=parent.recv()
        emit("model_service_started",**ready)
        if ready.get("type")!="ready":
            raise ServiceFailure(str(ready))
        result["service"]=ready
        if review:
            from .review_execution import ReviewExecution
            runner=ReviewExecution(ctx,parent,process,canceled,limits,emit)
        else:
            runner=Execution(ctx,parent,process,canceled,limits,emit)
        for index,(method,case) in enumerate(schedule):
            identity={"case_id":case["id"],"method":method}
            runner.active_method=method
            try:
                directory=output/method/case["id"] if review else output/case["id"]
                result["cases"][index]=runner.run_case(case,directory)
                result["c2c_executed"] |= result["cases"][index].get("c2c_executed",False)
            except Exception as exc:
                result["cases"][index]={"id":case["id"],"status":"execution_failed",
                    "error":{"type":type(exc).__name__,"message":str(exc)},
                    "decisions":getattr(runner,"current_decisions",[]),"tasks":list(runner.tasks.values()),
                    "model_service":runner.service_records,
                    "final_files":runner.workspace.snapshot() if hasattr(runner,"workspace") else None,
                    "observed_parent_wall_seconds":perf_counter()-runner.case_started if hasattr(runner,"case_started") else None}
                save()
                raise
            save()
            row=result["cases"][index]
            print(method,case["id"],row["status"],"delegations",row["delegations"],"completed",row["completed_goal"],
                  "information_block",row["declared_information_block_with_known_missing"],"wall",round(row["observed_parent_wall_seconds"],3),flush=True)
        result["status"]="completed"
    except Exception as exc:
        result.update(status="failed",error={"type":type(exc).__name__,"message":str(exc)})
        emit("run_failed",error=result["error"])
        raise
    finally:
        if runner:
            for tid in list(runner.live):
                runner.cancel(tid,"Runtime closing")
        if process.pid is not None:
            try:
                parent.send({"kind":"stop"})
            except (BrokenPipeError,EOFError,OSError):
                pass
            process.join(timeout=3)
            if process.is_alive():
                process.terminate(); process.join(timeout=3)
            if process.is_alive():
                process.kill(); process.join(timeout=3)
            emit("model_service_reaped",pid=process.pid,exitcode=process.exitcode,alive=process.is_alive())
            result["model_service_reaped"]=not process.is_alive()
            process.close()
        parent.close()
        result.update(observed_batch_wall_seconds=perf_counter()-started,event_logging_seconds=logging_seconds,
                      timing_scope="actual load, all parent decisions, OS workers, tool operations, canceled paid work and integrity audits; intervals overlap")
        (output/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
        log.close()
