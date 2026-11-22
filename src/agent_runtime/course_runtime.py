"""Autonomous scheduling-plan application over an immutable public OS snapshot.

Real Agent subprocesses, asynchronous supervision and versioned AR publications.
The GPU owner serves the actual inference. Snapshot business jobs are not run.
"""
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import shutil
import signal
from time import perf_counter, sleep

from .course_contract import check_plan, merge_proposal, PLAN_INTERFACE
from .course_scheduler import pick_ready
from .course_dag import dependency_status,has_agent_cycle
from .project import ROOT

TERMINAL = {"REPORTED", "SUCCEEDED", "FAILED", "CANCELLED", "NEEDS_INFORMATION"}


def _task_process(pipe, task_id, cgroup, directory):
    """Native task lifecycle proxy; it does not own the GPU models."""
    try:
        if cgroup:
            Path(cgroup, "cgroup.procs").write_text(str(os.getpid()))
        pipe.send({"event": "ready", "task_id": task_id, "pid": os.getpid()})
        result = pipe.recv()
        directory=Path(directory)
        directory.mkdir(parents=True,exist_ok=False)
        body=json.dumps(result,ensure_ascii=False,indent=2).encode('utf-8')
        (directory/"delivered-model-result.json").write_bytes(body)
        if "generated" in result:
            (directory/"report.txt").write_text(result["generated"]["output"],encoding="utf-8")
        result["process_owned_artifact"]={"relative_directory":directory.name,
            "sha256":hashlib.sha256(body).hexdigest(),"producer_pid":os.getpid()}
        pipe.send({"event": "delivered", "task_id": task_id, "result": result})
    finally:
        pipe.close()


def run_course(output_dir, *, method="c2c_hidden_analysis", public_filename="public.json",
               policy="fifo", max_decisions=10, max_tasks=3, fault_first_worker=False,
               cgroup_root=None, remote_config=None,remote_password=None):
    from .course_models import serve
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    public_path = ROOT/"data/c2c-core-os"/public_filename
    if public_filename != "public.json" and not public_filename.startswith("course-public-"):
        raise ValueError("Course runtime accepts reviewed public-only inputs; frozen probe files are diagnostic-only")
    raw_case = json.loads(public_path.read_text(encoding="utf-8"))
    case = {k:raw_case[k] for k in ("id","goal","public_background","current_state")}
    config = json.loads((ROOT/"configs/cloud.json").read_text())
    lock = json.loads((ROOT/"configs/cloud.models.lock.json").read_text())
    ctx = mp.get_context("spawn")
    requests, responses = ctx.Queue(), ctx.Queue()
    if remote_config:
        from .course_transport import remote_proxy
        gpu=ctx.Process(target=remote_proxy,args=(requests,responses,remote_config,remote_password,str(output)),name="course-gpu-bridge")
    else:
        gpu = ctx.Process(target=serve, args=(requests,responses,config,lock,str(output)), name="course-gpu-owner")
    start = perf_counter()
    acbs, live, calls, publications = {}, {}, {}, {}
    plan, contributions, observations, decisions, events = {}, [], [], [], []
    service, current_planner, last_action = {}, None, None
    status, need_plan, wait_for_event = "running", True, False
    next_task, injected = 1, False
    model_seconds, audit_seconds, model_load_seconds = 0.0, 0.0, 0.0
    released = set()
    source_hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in list((ROOT/"src/agent_runtime").glob("*.py"))+[ROOT/"scripts/run.py"]}
    for rel in source_hashes:
        dst=output/"source"/rel
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/rel,dst)
    last_persist = 0.0
    cgroot = Path(cgroup_root).resolve() if cgroup_root else None
    if cgroot and (not str(cgroot).startswith("/sys/fs/cgroup/") or not cgroot.is_dir()):
        raise ValueError("cgroup root must be an existing explicitly delegated project subtree")

    def event(kind, **fields):
        item = dict(event=kind, elapsed=perf_counter()-start, **fields)
        events.append(item)
        with (output/"events.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(item,ensure_ascii=False)+"\n")

    def submit(req):
        req["submitted_monotonic"] = perf_counter()
        calls[req["id"]] = {"kind": req["kind"], "publication": req.get("publication"), "submitted":perf_counter()}
        requests.put(req)

    def persist():
        snapshot = dict(status=status, method=method, policy=policy, public_sha256=hashlib.sha256(public_path.read_bytes()).hexdigest(),
            case_id=case["id"], goal=case["goal"], scope="Autonomous Agent scheduling-plan artifact; immutable public business snapshot, no business execution claim",
            started_pid=os.getpid(), gpu_service=service, cgroup_enabled=bool(cgroot), model_pins=lock,
            acbs=list(acbs.values()), decisions=decisions, plan=plan, final_check=check_plan(case,plan,complete=True),
            contributions=contributions, worker_observations=observations, wall_seconds=perf_counter()-start,
            model_call_seconds=model_seconds, model_load_seconds=model_load_seconds, integrity_logging_seconds=audit_seconds,
            delegation_count=len(acbs), successful_worker_reports=sum(a["state"]=="SUCCEEDED" for a in acbs.values()),
            delivered_worker_reports=sum(a["state"] in {"REPORTED","SUCCEEDED"} for a in acbs.values()),
            accepted_worker_reports=sum(c["actor"]=="Worker" for c in contributions),
            actual_fused_consumptions=sum(bool(a.get("worker_consumed_fused_cache")) for a in acbs.values()),
            source_sha256=source_hashes)
        tmp = output/"state.tmp"
        tmp.write_text(json.dumps(snapshot,ensure_ascii=False,indent=2),encoding="utf-8")
        tmp.replace(output/"result.json")
        (output/"plan.json").write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding="utf-8")

    def finish_task(tid, state, **facts):
        nonlocal need_plan, wait_for_event
        a = acbs[tid]
        if a["state"] in TERMINAL:
            return
        a.update(state=state, finished_elapsed=perf_counter()-start, **facts)
        event("task_terminal", task_id=tid, state=state, facts=facts)
        observations.append({"task_id":tid, "state":state, **facts})
        need_plan, wait_for_event = True, False

    def dependencies(task):
        jobs = {j["id"]:j for j in case["public_background"]["tasks"]}
        return dependency_status(task,acbs,jobs)

    gpu.start()
    remote_password=None
    event("gpu_service_started", pid=gpu.pid)
    try:
        while status == "running":
            if (not service and perf_counter()-start > 600) or perf_counter()-start > 3600:
                status = "runtime_deadline_exceeded"
                break
            if any(perf_counter()-c["submitted"] > 900 for c in calls.values()):
                status = "model_request_deadline_exceeded"
                break
            # Collect GPU completion without blocking process supervision.
            for _ in range(100):
                try:
                    reply = responses.get_nowait()
                except queue.Empty:
                    break
                if reply["kind"] == "ready":
                    service = reply
                    model_load_seconds = reply["load_seconds"]
                    event("gpu_service_ready", **{k:v for k,v in reply.items() if k!="kind"})
                    continue
                if reply["kind"] == "fatal":
                    event("gpu_fatal", error=reply["error"])
                    status = "model_service_failed"
                    break
                if reply["kind"] == "started":
                    if reply["id"] in acbs and acbs[reply["id"]]["state"] not in TERMINAL:
                        acbs[reply["id"]].update(state="MODEL_RUNNING",model_queue_seconds=reply["queue_seconds"])
                    event("model_started",call_id=reply["id"],queue_seconds=reply["queue_seconds"])
                    continue
                info = calls.pop(reply["id"])
                model_seconds += reply["wall_seconds"]
                audit_seconds += reply["integrity_logging_seconds"]
                event("model_completed", call_id=reply["id"], raw_record=reply["raw_record"], error=reply.get("error"))
                if info["kind"] == "worker":
                    tid = reply["id"]
                    if tid in live and acbs[tid]["state"] not in TERMINAL:
                        live[tid]["pipe"].send(reply)
                    else:
                        event("orphaned_model_result", task_id=tid, raw_record=reply["raw_record"], cost_counted=True)
                    continue
                current_planner = None
                if "error" in reply:
                    decisions.append({"raw_record":reply["raw_record"], "error":reply["error"]})
                    observations.append({"planner_failure":reply["error"]})
                    need_plan = True
                    continue
                action = reply["action"]
                last_action = action
                decisions.append({"action":action, "raw_record":reply["raw_record"]})
                event("planner_action", action=action)
                name = action["action"]
                need_plan = True
                if name == "delegate":
                    publication = reply["publication"]
                    publications[publication] = []
                    # All IDs exist before resolving same-batch Agent dependencies.
                    for task in action["tasks"]:
                        tid = f"a{next_task:03d}"
                        next_task += 1
                        acbs[tid] = dict(id=tid, task=task, state="PENDING", publication=publication,
                            created_elapsed=perf_counter()-start,ready_sequence=next_task, virtual_service=0.0, attribution="Worker", model_owner_pid=service["pid"])
                        publications[publication].append(tid)
                    # Reject cycles without manufacturing an execution order.
                    for tid in publications[publication]:
                        a = acbs[tid]
                        blocked, invalid = dependencies(a["task"])
                        cycle = has_agent_cycle(tid,acbs)
                        if invalid or cycle:
                            finish_task(tid,"FAILED", error="invalid_or_cyclic_dependencies", invalid=invalid)
                        else:
                            a["state"] = "BLOCKED" if blocked else "READY"
                elif name == "report":
                    verdict = check_plan(case,action["report"])
                    evidence_ids=action.get("evidence_task_ids",[])
                    invalid=[tid for tid in evidence_ids if tid not in acbs or acbs[tid]["state"] not in {"REPORTED","SUCCEEDED"}]
                    if invalid:
                        verdict={"accepted":False,"errors":["evidence_task_ids must reference delivered reports"],"invalid_task_ids":invalid}
                    if verdict["accepted"]:
                        plan = action["report"]
                        contributions.append({"actor":"Planner", "kind":"takeover" if acbs else "self_execution", "decision":len(decisions), "report":plan,"explicitly_cited_worker_evidence":evidence_ids,"causal_worker_contribution_proven":False})
                    observations.append({"producer":"Planner","decision_id":reply["id"],"actual_proposal":action["report"],"planner_report_check":verdict})
                elif name == "accept":
                    a = acbs.get(action["task_id"])
                    if not a or a["state"] not in {"REPORTED","SUCCEEDED"} or not isinstance(a.get("report"),dict):
                        observations.append({"accept_failed":"task has no validated report", "task_id":action["task_id"]})
                    else:
                        merged, verdict = merge_proposal(case,plan,a["report"])
                        if verdict["accepted"]:
                            plan = merged
                            contributions.append({"actor":"Worker", "task_id":a["id"], "report":a["report"], "decision":len(decisions)})
                        observations.append({"accept_check":verdict,"task_id":a["id"]})
                elif name == "cancel":
                    tid = action["task_id"]
                    if tid in acbs:
                        finish_task(tid,"CANCELLED", reason=action["reason"])
                    else:
                        observations.append({"cancel_failed":"unknown task", "task_id":tid})
                elif name == "wait":
                    if any(a["state"] not in TERMINAL for a in acbs.values()):
                        need_plan, wait_for_event = False, True
                    else:
                        observations.append({"wait_result":"no unfinished Agent tasks"})
                elif name == "finish":
                    verdict = check_plan(case,plan,complete=True)
                    unfinished = [a["id"] for a in acbs.values() if a["state"] not in TERMINAL]
                    if action["cancel_pending"]:
                        for tid in unfinished:
                            finish_task(tid,"CANCELLED",reason="explicit Planner finish cancellation")
                        unfinished = []
                    if action["status"] == "needs_information":
                        status = "needs_information"
                    elif verdict["accepted"] and not unfinished:
                        status = "completed"
                    else:
                        observations.append({"finish_check":verdict,"unfinished_agent_tasks":unfinished})

            # Task processes have genuine PIDs, exit codes and pipes; GPU compute is attributed separately.
            for tid, owner in list(live.items()):
                proc, pipe = owner["process"], owner["pipe"]
                a = acbs[tid]
                try:
                    # Actual Linux task-proxy CPU/RSS, distinct from shared GPU owner.
                    stat = Path(f"/proc/{proc.pid}/stat").read_text().rsplit(")",1)[1].split()
                    a["process_cpu_seconds"] = (int(stat[11])+int(stat[12]))/os.sysconf("SC_CLK_TCK")
                    a["process_rss_bytes"] = int(stat[21])*os.sysconf("SC_PAGE_SIZE")
                    a["process_peak_rss_bytes"] = max(a.get("process_peak_rss_bytes",0),a["process_rss_bytes"])
                except (OSError,ValueError,IndexError):
                    pass
                if a["state"] in TERMINAL and proc.is_alive():
                    proc.terminate()
                if pipe.poll():
                    try:
                        msg = pipe.recv()
                    except EOFError:
                        msg = None
                    if msg and msg["event"] == "ready" and a["state"] not in TERMINAL:
                        event("worker_process_ready",task_id=tid,pid=msg["pid"])
                        if fault_first_worker and not injected:
                            injected = True
                            os.kill(proc.pid,signal.SIGKILL)
                            event("fault_injected",task_id=tid,pid=proc.pid,fault_kind="SIGKILL_before_GPU_submission")
                        else:
                            deps = [{"task_id":d[6:],"raw_output":acbs[d[6:]]["raw_output"],"report":acbs[d[6:]].get("report"),"semantic_verified":acbs[d[6:]].get("semantic_verified",False),"dependency_condition":"actual_report_delivered"}
                                    for d in a["task"]["depends_on"] if d.startswith("agent:")]
                            control = dict(task_id=tid,snapshot_sha256=hashlib.sha256(public_path.read_bytes()).hexdigest(),
                                snapshot_version=0,dependency_artifacts=deps,
                                explicit_channel="task contract, status, dependency artifacts only")
                            submit(dict(kind="worker",id=tid,publication=a["publication"],task=a["task"],method=method,control=control))
                            a["state"]="WAIT_MODEL"
                    elif msg and msg["event"] == "delivered" and a["state"] not in TERMINAL:
                        r = msg["result"]
                        a["model_record"] = r["raw_record"]
                        a["process_owned_artifact"] = r.get("process_owned_artifact")
                        a["virtual_service"] += r["wall_seconds"]
                        if "error" in r:
                            finish_task(tid,"FAILED",error=r["error"])
                        else:
                            generated = r["generated"]
                            a["worker_consumed_fused_cache"] = generated.get("worker_consumed_fused_cache",False)
                            raw_output=generated["output"]
                            a["raw_output"]=raw_output
                            try:
                                decoded = json.loads(raw_output)
                            except (ValueError,TypeError):
                                decoded=None
                            try:
                                if not str(raw_output).strip():
                                    finish_task(tid,"FAILED",error="empty_worker_output")
                                    continue
                                if not isinstance(decoded,dict):
                                    finish_task(tid,"REPORTED",report=raw_output,raw_output=raw_output,transport_validation="original_JSON_wrapper_absent",assignment_completed="unverified",semantic_verified=False)
                                    continue
                                if decoded.get("action") == "need_info" and str(decoded.get("question","")).strip():
                                    finish_task(tid,"NEEDS_INFORMATION",question=decoded["question"])
                                elif decoded.get("action") == "report":
                                    report=decoded.get("report")
                                    verdict = check_plan(case,report) if isinstance(report,dict) else None
                                    finish_task(tid,"REPORTED",report=report,raw_output=raw_output,transport_validation="original_report_wrapper_present",assignment_completed="unverified",semantic_verified=False,plan_object_check=verdict)
                                else:
                                    finish_task(tid,"REPORTED",report=decoded,raw_output=raw_output,transport_validation="original_report_wrapper_absent",assignment_completed="unverified",semantic_verified=False)
                            except (ValueError,TypeError) as exc:
                                finish_task(tid,"FAILED",error=str(exc),raw_output=generated["output"])
                if not proc.is_alive():
                    proc.join()
                    a["exitcode"] = proc.exitcode
                    if a["state"] not in TERMINAL:
                        finish_task(tid,"FAILED",error="worker_process_exit",exitcode=proc.exitcode)
                    pipe.close()
                    if owner["cgroup"]:
                        a['cgroup_resource_events']={name:(owner['cgroup']/name).read_text() for name in ['cpu.stat','memory.events','memory.peak','pids.current']}
                        owner["cgroup"].rmdir()
                    del live[tid]
                elif perf_counter()-owner["started"] > 600:
                    finish_task(tid,"FAILED",error="worker_lifecycle_timeout_600s")

            # Resolve the dynamic Agent DAG using real terminal results.
            for tid,a in acbs.items():
                if a["state"] == "BLOCKED":
                    blocked,_ = dependencies(a["task"])
                    if not blocked:
                        a["state"] = "READY"
                        event("dependency_ready",task_id=tid)
                    elif any(d.startswith("agent:") and acbs[d[6:]]["state"] in TERMINAL-{"SUCCEEDED","REPORTED"} for d in blocked):
                        finish_task(tid,"FAILED",error="dependency_failed",blocked=blocked)
            while len(live) < max_tasks and status == "running" and (a:=pick_ready(acbs.values(),policy)) is not None:
                parent,child = ctx.Pipe()
                cg = cgroot/a["id"] if cgroot else None
                if cg:
                    cg.mkdir()
                    (cg/"memory.max").write_text(str(512*1024*1024))
                    (cg/"pids.max").write_text("16")
                    (cg/"cpu.max").write_text("100000 100000")
                proc = ctx.Process(target=_task_process,args=(child,a["id"],str(cg) if cg else None,str(output/("task-"+a["id"]))),name="course-"+a["id"])
                proc.start()
                child.close()
                a.update(state="RUNNING",pid=proc.pid,started_elapsed=perf_counter()-start,cgroup=str(cg) if cg else None)
                live[a["id"]] = dict(process=proc,pipe=parent,started=perf_counter(),cgroup=cg)
                event("task_started",task_id=a["id"],pid=proc.pid,cgroup=a["cgroup"])

            # Publication lifetimes include pending GPU requests, even after a proxy dies.
            for pub,tids in publications.items():
                if pub not in released and all(acbs[t]["state"] in TERMINAL for t in tids) and not any(c["publication"]==pub for c in calls.values()):
                    requests.put({"kind":"release","publication":pub})
                    released.add(pub)
                    event("publication_released",publication=pub)
            if status != "running":
                break
            if not gpu.is_alive():
                status = "model_service_exit"
                break
            if service and need_plan and current_planner is None:
                if len(decisions) >= max_decisions:
                    status = "decision_budget_exhausted"
                else:
                    current_planner = f"p{len(decisions)+1:03d}"
                    context = dict(plan_interface=PLAN_INTERFACE,plan=plan,plan_check=check_plan(case,plan,complete=True),
                        agent_tasks=list(acbs.values()),
                        worker_observation=[o for o in observations if "task_id" in o and "state" in o],
                        runtime_observations=observations,last_action=last_action,
                        next_available_agent_ids=[f"a{i:03d}" for i in range(next_task,next_task+max_tasks)],
                        remaining_decisions=max_decisions-len(decisions),dispatch_boundary=len(events))
                    submit(dict(kind="planner",id=current_planner,case=case,context=context,max_tasks=max_tasks))
                    need_plan=False
            if wait_for_event and not live and not any(a["state"]=="READY" for a in acbs.values()):
                observations.append({"wait_result":"only blocked tasks remain", "blocked":[a["id"] for a in acbs.values() if a["state"]=="BLOCKED"]})
                need_plan,wait_for_event=True,False
            if perf_counter()-last_persist > 1:
                persist()
                last_persist=perf_counter()
            sleep(.05)
    except BaseException as exc:
        status = "runtime_exception"
        event("runtime_exception",error=repr(exc))
        raise
    finally:
        for tid,owner in live.items():
            if owner["process"].is_alive():
                owner["process"].terminate()
            owner["process"].join(5)
            if owner["process"].is_alive():
                owner["process"].kill()
                owner["process"].join(5)
            owner["pipe"].close()
            acbs[tid]["exitcode"] = owner["process"].exitcode
            if acbs[tid]["state"] not in TERMINAL:
                finish_task(tid,"CANCELLED",reason="runtime terminated; not success")
            if owner["cgroup"]:
                try:
                    owner["cgroup"].rmdir()
                except OSError as exc:
                    event("cgroup_cleanup_failed",task_id=tid,error=str(exc))
        requests.put({"kind":"stop"})
        # This only stops our model subprocess, never shuts down the authorized server.
        cleanup_deadline = perf_counter()+60
        stopped = False
        while (gpu.is_alive() or calls) and perf_counter()<cleanup_deadline:
            try:
                r = responses.get(timeout=.2)
            except queue.Empty:
                continue
            if r.get("kind")=="result":
                calls.pop(r["id"],None)
                model_seconds += r["wall_seconds"]
                audit_seconds += r["integrity_logging_seconds"]
                event("late_model_completed",call_id=r["id"],raw_record=r["raw_record"],cost_counted=True)
            elif r.get("kind")=="stopped":
                stopped=True
            elif r.get("kind")=="remote_exit":
                event("remote_owner_exit_confirmed",exitcode=r["exitcode"])
        if gpu.is_alive():
            gpu.terminate()
            gpu.join(5)
        if gpu.is_alive():
            gpu.kill()
        gpu.join(5)
        if calls:
            event("unresolved_model_costs",call_ids=list(calls),cost_status="unknown_incomplete_not_zero")
        event("runtime_finished",status=status)
        persist()
    return json.loads((output/"result.json").read_text(encoding="utf-8"))
