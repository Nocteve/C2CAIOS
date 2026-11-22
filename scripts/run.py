"""Run the autonomous application or explicit frozen diagnostics; never load references."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--autonomy", action="store_true", help="Frozen eight-parent autonomous delegation engineering run")
    parser.add_argument('--autonomy-protocol', type=Path, help='Explicit hash-locked configuration engineering protocol inside the project')
    parser.add_argument('--worker-capability',action='store_true',help='Frozen independent ledger Worker-component capability and necessary small adaptation; not autonomy or C2C')
    parser.add_argument('--worker-capability-protocol',type=Path,help='Project-local preregistered Worker-component protocol')
    parser.add_argument("--communication-method",choices=('text_full','text_brief','c2c_shared'),help="Configuration autonomy: actual Worker background communication; delegation remains optional")
    parser.add_argument("--planner-tools", action="store_true", help="Autonomy/review: native Qwen tool signatures/calls; free Planner decode and strict validation")
    parser.add_argument("--worker-tools", action="store_true", help="Autonomy: Worker-owned native tools and actual tool responses, free decoding")
    parser.add_argument("--worker-context-mode", choices=('planner_transcript','independent_session','semantic_session'),
                        help="Configuration autonomy: independent text Worker session or legacy Planner transcript")
    parser.add_argument("--planner-tool-feedback", action="store_true", help="With --planner-tools: return actual prior execution and current observations as native tool feedback")
    parser.add_argument("--planner-thinking", action="store_true", help="With --planner-tools: native Qwen thinking and recommended sampling, exact original KV retained")
    parser.add_argument("--planner-analysis", action="store_true", help="Autonomy/review only: real bounded analysis followed by action on the same KV")
    parser.add_argument("--code-review", action="store_true", help="One real scoped review; three independent autonomous communication arms")
    parser.add_argument("--review-method", choices=('text_full','text_brief','c2c_shared'), help="With --code-review, run one explicitly recorded arm")
    parser.add_argument("--review-data-dir", type=Path, help="With --code-review, an explicit hash-locked public review packet; never a reference-answer file")
    parser.add_argument("--arc-native-calibration", action="store_true", help="Diagnostic only: missing native receiver arm on 128 frozen historical C2C inputs")
    parser.add_argument("--fuser-readiness", action="store_true", help="Offline one-example gradient audit; no optimizer or saved weights")
    parser.add_argument("--fuser-pilot", action="store_true", help="One preregistered scalar-head training pass and held-out handoff comparison")
    parser.add_argument("--scalar-heads-only", action="store_true", help="With --fuser-readiness, audit only the frozen scalar-head allowlist")
    parser.add_argument("--qasper-dev", action="store_true", help="Historical fixed six-paper development workflow; not the autonomous application")
    parser.add_argument("--planner-call-diagnostic", action="store_true", help="Six offline native next-action calls; no Worker, edits or C2C")
    parser.add_argument("--planner-parity", action="store_true", help="All-eight fixed real-history numerical and full-action replay audit")
    parser.add_argument("--execution-calibration", action="store_true", help="One fixed eight-pair native short-context execution diagnosis; not a communication benchmark")
    parser.add_argument("--communication-probe", action="store_true", help="Fixed actual first-delegation full-text/handoff/C2C mechanism replay")
    parser.add_argument("--c2c-core", action="store_true", help="One OS-runtime Planner decision and native Worker C2C handoff")
    parser.add_argument("--c2c-core-rich", action="store_true", help="OS-runtime rich Planner planning note followed by full/T2T/C2C Worker comparison")
    parser.add_argument("--c2c-core-loop-rich", action="store_true", help="Rich Planner/Worker comparison followed by per-arm Planner feedback decision")
    parser.add_argument("--c2c-core-rich-hidden", action="store_true", help="Rich Planner comparison plus hidden-context arms: Worker reads only the short command text while selected Planner KV is fused as hidden prefix context")
    parser.add_argument("--communication-extension", action="store_true", help="Explicit multi-task extension of the rich-hidden diagnostic; original defaults unchanged")
    parser.add_argument("--c2c-core-ablate", action="store_true", help="A3 drop-one receiver-utility ablation over the analysis selection (implies rich planning; no gold answer used)")
    parser.add_argument("--c2c-core-multi", action="store_true", help="OS-runtime Planner handoff with up to --max-tasks actual Worker tasks sharing one C2C cache")
    parser.add_argument("--c2c-core-acb", action="store_true", help="B1: ACB-scheduled multi-task C2C (DAG deps, GPU-budget admission, heartbeat, bounded repairs; hidden-analysis arm through the public OS gate)")
    parser.add_argument("--c2c-core-acb-isolated", action="store_true", help="B1 isolated variant: every Worker task runs as a real OS child process (cgroup/rlimit limits) consuming a published fused KV payload")
    parser.add_argument("--sched-fair", action="store_true", help="With --c2c-core-acb-isolated: CFS min-vruntime fair scheduling + wait-for deadlock probe (os_sched.FairScheduler)")
    parser.add_argument("--cgroups-quota", type=str, default="", help="With --c2c-core-acb-isolated: JSON dict enabling the simulated cgroup v2 controller, e.g. {\"tokens\": 50, \"wall_s\": 10, \"memory_mib\": 512, \"slice_pids_max\": 8}; empty disables")
    parser.add_argument("--payload-mmap", action="store_true", help="With --c2c-core-acb-isolated: publish the worker cache as a raw bin the child maps with np.memmap (host zero-copy load) instead of torch.load")
    parser.add_argument("--payload-pool-capacity", type=int, default=2, help="With --c2c-core-acb-isolated: LRU pool capacity for resident published payloads; excess is swap-out, re-execution of an evicted payload records a page fault")
    parser.add_argument("--max-tasks", type=int, default=2, help="Task ceiling for --c2c-core-multi (default 2; the Planner may delegate fewer)")
    parser.add_argument("--rich-planning", action="store_true", help="Planner writes a rich analysis note before the action (multi mode: answers get computed before dispatch)")
    parser.add_argument("--arith-scaffold", action="store_true", help="With --c2c-core-rich-hidden: force the Planner note to show explicit arithmetic for every decision value (analysis budget 320->512)")
    parser.add_argument("--public-filename", default=None, help="Optional public scenario file under data/c2c-core-os/ (defaults: public.json, multi uses public-multi.json)")
    parser.add_argument("--worker-recovery", action="store_true", help="Execute all original recorded Worker assignments through the real Worker process, old/new interfaces")
    parser.add_argument("--worker-recovery-protocol",type=Path,help="Explicit frozen protocol for --worker-recovery")
    parser.add_argument("--fault-check", action="store_true", help="First fixed paper, terminate one OS worker per arm")
    parser.add_argument("--mechanism-pilot", action="store_true", help="Fixed first two papers, four cache conditions, no repair or successors")
    parser.add_argument("--source-diagnostic", action="store_true", help="Four fixed source-side retrievals; never Worker comparison scores")
    parser.add_argument("--order-study", action="store_true", help="Frozen new development papers, both block orders, three multi-agent methods with repeated timing")
    parser.add_argument("--catalog-diagnostic", action="store_true", help="Fixed 24 retrievals: original versus reversed valid-ID catalog; no successors or repairs")
    parser.add_argument("--interface-study", action="store_true", help="All 16 development papers, default paper-first order, exact array codec, three arms and repeats")
    args = parser.parse_args()
    if args.worker_capability_protocol and not args.worker_capability:
        parser.error('--worker-capability-protocol requires --worker-capability')
    if args.worker_capability and not args.worker_capability_protocol:
        parser.error('--worker-capability requires its frozen protocol')
    if args.autonomy_protocol and not args.autonomy:
        parser.error('--autonomy-protocol requires --autonomy')
    if args.worker_recovery_protocol and not args.worker_recovery:
        parser.error('--worker-recovery-protocol requires --worker-recovery')
    if args.communication_method and not args.autonomy:
        parser.error('--communication-method requires --autonomy')
    if args.planner_tool_feedback and not args.planner_tools:
        parser.error('--planner-tool-feedback requires --planner-tools')
    if args.worker_tools and (not args.planner_tools or args.code_review):
        parser.error('--worker-tools requires --planner-tools in the configuration application')
    if args.communication_method == 'c2c_shared' and not args.planner_tools:
        parser.error('c2c_shared requires --planner-tools so the Worker receives an explicit role boundary')
    if args.worker_context_mode and (not args.autonomy or not args.worker_tools):
        parser.error('--worker-context-mode requires --autonomy --worker-tools')
    if args.worker_context_mode in {'independent_session','semantic_session'} and args.communication_method == 'c2c_shared':
        parser.error('Independent C2C has no validated source/receiver position mapping')
    if args.planner_thinking and not args.planner_tools:
        parser.error('--planner-thinking requires --planner-tools')
    if args.review_method and not args.code_review:
        parser.error('--review-method requires --code-review')
    if args.review_data_dir and not args.code_review:
        parser.error('--review-data-dir requires --code-review')
    if args.scalar_heads_only and not args.fuser_readiness:
        parser.error("--scalar-heads-only requires --fuser-readiness")
    modes=(args.worker_capability,args.worker_recovery,args.fuser_pilot,args.fuser_readiness,args.arc_native_calibration,args.code_review,args.communication_probe,args.c2c_core,args.c2c_core_rich,args.c2c_core_loop_rich,args.c2c_core_rich_hidden,args.c2c_core_ablate,args.c2c_core_multi,args.c2c_core_acb,args.c2c_core_acb_isolated,args.execution_calibration,args.planner_parity,args.planner_call_diagnostic,args.qasper_dev,args.fault_check,args.mechanism_pilot,args.source_diagnostic,args.order_study,args.catalog_diagnostic,args.interface_study)
    if sum((args.autonomy,*modes)) > 1:
        parser.error("Recovery, mechanism and source diagnosis are separate runs")
    if args.planner_tools and ((any(modes) and not args.code_review) or args.planner_analysis):
        parser.error('--planner-tools applies only to autonomous engineering/review without added analysis')
    if args.planner_analysis and any(modes) and not args.code_review:
        parser.error("--planner-analysis applies only to autonomous application or code review")
    if args.autonomy or not any(modes):
        from agent_runtime.autonomy import run_autonomy
        return run_autonomy(args.output_dir,planner_analysis=args.planner_analysis,planner_tools=args.planner_tools,
                            planner_tool_feedback=args.planner_tool_feedback,planner_thinking=args.planner_thinking,worker_tools=args.worker_tools,
                            communication_method=args.communication_method,worker_context_mode=args.worker_context_mode,protocol_path=args.autonomy_protocol)
    if args.code_review:
        from agent_runtime.autonomy import run_autonomy
        return run_autonomy(args.output_dir,review=True,planner_analysis=args.planner_analysis,review_method=args.review_method,
                            planner_tools=args.planner_tools,planner_tool_feedback=args.planner_tool_feedback,
                            review_data_dir=args.review_data_dir,planner_thinking=args.planner_thinking)
    if args.worker_recovery:
        from agent_runtime.worker_recovery import run_worker_recovery
        return run_worker_recovery(args.output_dir,args.worker_recovery_protocol)
    if args.worker_capability:
        from agent_runtime.worker_capability import run_worker_capability
        return run_worker_capability(args.output_dir,args.worker_capability_protocol)
    if args.arc_native_calibration:
        from agent_runtime.diagnostic import run_arc_native_calibration
        return run_arc_native_calibration(args.output_dir)
    if args.fuser_readiness:
        from agent_runtime.fuser_readiness import run_fuser_readiness
        return run_fuser_readiness(args.output_dir, scalar_heads_only=args.scalar_heads_only)
    if args.fuser_pilot:
        from agent_runtime.fuser_pilot import run_fuser_pilot
        return run_fuser_pilot(args.output_dir)
    if args.planner_call_diagnostic:
        from agent_runtime.planner_call_diagnostic import run_planner_call_diagnostic
        return run_planner_call_diagnostic(args.output_dir)
    if args.planner_parity:
        from agent_runtime.planner_call_diagnostic import run_planner_parity
        return run_planner_parity(args.output_dir)
    if args.execution_calibration:
        from agent_runtime.execution_calibration import run_execution_calibration
        return run_execution_calibration(args.output_dir)
    if args.communication_probe:
        from agent_runtime.communication_probe import run_communication_probe
        return run_communication_probe(args.output_dir)
    if args.c2c_core:
        from agent_runtime.c2c_core import run_c2c_core
        return run_c2c_core(args.output_dir, public_filename=args.public_filename or "public.json")
    if args.c2c_core_rich:
        from agent_runtime.c2c_core import run_c2c_core
        return run_c2c_core(args.output_dir, public_filename=args.public_filename or "public.json", rich_planning=True)
    if args.c2c_core_ablate:
        from agent_runtime.c2c_core import run_c2c_core
        return run_c2c_core(args.output_dir, public_filename=args.public_filename or "public.json",
                            rich_planning=True, arith_scaffold=args.arith_scaffold, hidden_ablate=True)
    if args.c2c_core_loop_rich:
        from agent_runtime.c2c_core import run_c2c_core
        return run_c2c_core(args.output_dir, public_filename=args.public_filename or "public.json", rich_planning=True, feedback_loop=True, arith_scaffold=args.arith_scaffold)
    if args.c2c_core_rich_hidden:
        from agent_runtime.c2c_core import run_c2c_core
        count = args.max_tasks if args.communication_extension else 1
        return run_c2c_core(args.output_dir, public_filename=args.public_filename or "public.json", rich_planning=True, hidden_arms=True, arith_scaffold=args.arith_scaffold,
                            max_tasks=count, planner_action_tokens=(384 * count if args.communication_extension and count > 1 else None))
    if args.c2c_core_multi:
        from agent_runtime.c2c_core import run_c2c_core
        return run_c2c_core(args.output_dir, public_filename=args.public_filename or "public-multi.json",
                            max_tasks=args.max_tasks, rich_planning=args.rich_planning,
                            arith_scaffold=args.arith_scaffold)
    if args.c2c_core_acb:
        from agent_runtime.c2c_core import run_c2c_core_acb
        return run_c2c_core_acb(args.output_dir, public_filename=args.public_filename or "public-multi.json",
                                max_tasks=args.max_tasks, arith_scaffold=True)
    if args.c2c_core_acb_isolated:
        from agent_runtime.c2c_core import run_c2c_core_acb_isolated
        sim_quota = json.loads(args.cgroups_quota) if args.cgroups_quota else None
        return run_c2c_core_acb_isolated(args.output_dir, public_filename=args.public_filename or "public-multi.json",
                                         max_tasks=args.max_tasks, arith_scaffold=True,
                                         sched_policy="fair" if args.sched_fair else "priority",
                                         sim_quota=sim_quota,
                                         payload_mmap=args.payload_mmap,
                                         payload_pool_capacity=args.payload_pool_capacity)
    if args.order_study:
        from agent_runtime.comparison import run_order_study
        return run_order_study(args.output_dir)
    if args.catalog_diagnostic:
        from agent_runtime.catalog_diagnostic import run_catalog_diagnostic
        return run_catalog_diagnostic(args.output_dir)
    if args.interface_study:
        from agent_runtime.comparison import run_order_study
        return run_order_study(args.output_dir, interface_v2=True)
    diagnostic = args.mechanism_pilot or args.source_diagnostic
    data = ROOT / "data/qasper-dev-six/public.json"
    manifest = json.loads(data.with_name("manifest.json").read_text())
    assert hashlib.sha256(data.read_bytes()).hexdigest() == manifest["public_sha256"]
    papers = json.loads(data.read_text(encoding="utf-8"))
    if args.fault_check:
        papers = papers[:1]
    if diagnostic:
        papers = papers[:2]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / "configs/cloud.json").read_text())
    lock = json.loads((ROOT / "configs/cloud.models.lock.json").read_text())
    from agent_runtime.models import ModelHost, METHODS
    from agent_runtime.runtime import execute
    run_started = perf_counter()
    result = {"scope": "process recovery" if args.fault_check else "six frozen development papers, twelve real questions",
              "manifest": manifest, "method_names": list(METHODS),
              "expected_paper_ids": [p["id"] for p in papers], "papers": [], "status": "running"}
    if args.mechanism_pilot:
        from agent_runtime.diagnostic import CONDITIONS, expected_cases, commands, numerical_audit, run_cases
        result.update(scope="fixed 2-paper 4-question mechanism diagnosis; not a system benchmark",
                      kind="mechanism_pilot", method_names=list(CONDITIONS), cases=expected_cases(papers),
                      recovery_policy="none", content_takeovers=0, answer_successors=False)
    if args.source_diagnostic:
        from agent_runtime.diagnostic import source_cases, run_source_cases
        result.update(scope="four fixed source-side retrievals; capability diagnosis, not a communication baseline",
                      kind="source_diagnostic", method_names=["source_4b"], cases=source_cases(papers),
                      recovery_policy="none", content_takeovers=0, answer_successors=False)
    log = (args.output_dir / "events.jsonl").open("x", encoding="utf-8")
    identity = {}
    def emit(kind, **fields):
        log.write(json.dumps({"type": kind, **identity, **fields}, ensure_ascii=False, allow_nan=False) + "\n")
        log.flush()
    sources = [*sorted((ROOT / "src/agent_runtime").glob("*.py")), Path(__file__).resolve()]
    emit("metadata", config=config, models=lock, manifest=manifest,
         source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
         worker_limit=0 if diagnostic else 2, worker_memory_limit_bytes=256*1024**2,
         gpu_owner_process=True, separate_model_service_process=False,
         cgroup_enabled=False, schema_constraints="Planner dispatch only; all workers decode freely and actual outputs are validated",
         reference_answers_accessible=False, methods=result["method_names"],
         recovery_policy="identical infrastructure retry only" if args.fault_check else "none",
         content_takeover_allowed=False, expected_cases=result.get("cases"))
    emit("output_contract", evidence="object_v1" if diagnostic else "evidence_ids_v2")
    try:
        host = ModelHost(config, lock)
        import torch, transformers
        emit("environment", torch=str(torch.__version__), transformers=transformers.__version__,
             gpu=torch.cuda.get_device_name(), python=sys.version)
        for paper in papers:
            identity = {"paper_id": paper["id"]}
            row = {"id": paper["id"], "title": paper["title"], "methods": {}}
            result["papers"].append(row)
            try:
                plan = host.prepare(paper, emit, mechanism=diagnostic)
                if args.source_diagnostic:
                    row["planning_seconds"] = host.planning_seconds
                    row["preparation_costs"] = host.costs
                    run_source_cases(host, paper, plan, result["cases"], emit)
                    continue
                if args.mechanism_pilot:
                    row["planning_seconds"] = host.planning_seconds
                    row["preparation_costs"] = host.costs
                    if paper["id"] == papers[0]["id"]:
                        audit = numerical_audit(host, commands(paper, plan)[paper["questions"][0]["id"]])
                        result["numerical_audit"] = audit
                        emit("numerical_audit", **audit)
                        if not audit["passed"]:
                            raise RuntimeError("Numerical comparison differs; preserve evidence and stop before pilot")
                    run_cases(host, paper, plan, result["cases"], emit)
                    continue
                for method in METHODS:
                    identity = {"paper_id": paper["id"], "method": method}
                    if method not in host.payloads:
                        row["methods"][method] = {"failed": True, "reason": "Incomplete actual text handoff",
                            "attributed_seconds": host.planning_seconds + host.costs[method]}
                        continue
                    execution = execute(paper, plan, host, method, emit, args.fault_check)
                    execution["attributed_seconds"] = host.planning_seconds + host.costs[method] + execution["wall_seconds"]
                    row["methods"][method] = execution
                    emit("context_audit", **host.audit())
                    print(paper["id"], method, "answers", sum(not r.get("failed") for r in execution["report"]),
                          "repairs", execution["repair_nodes"], flush=True)
            except Exception as exc:
                row["error"] = {"type": type(exc).__name__, "message": str(exc)}
                emit("paper_failed", **row["error"])
                print("FAILED", paper["id"], row["error"], flush=True)
                if diagnostic:
                    raise
        del host
        result["status"] = "completed"
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        result["status"] = "failed"
        raise
    finally:
        result["wall_seconds_including_loading_and_audits"] = perf_counter() - run_started
        result["timing_scope"] = ("mechanism diagnostic: source planning, native prefill, scope, fusion and worker model service reported separately; no OS workers, handoff, repair or answer; numerical audit and hashes are diagnostic overhead; fixed order, not speed evidence"
            if diagnostic else "attribution includes common planning, method preparation, OS worker startup, tools, scheduling and repair; excludes loading and integrity audits; development, fixed method order")
        if args.source_diagnostic:
            result["timing_scope"] = "source capability diagnosis: real planning and four source task services; shared preparation also constructs unused receiver/fused caches for identity audit; not a deployment latency comparison"
        (args.output_dir / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        emit("run_finished", status=result["status"])
        log.close()


if __name__ == "__main__":
    main()
