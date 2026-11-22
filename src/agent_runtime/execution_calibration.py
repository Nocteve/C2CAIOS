"""One fixed paired execution diagnosis; not an autonomous or C2C benchmark."""
import hashlib
import json
from pathlib import Path
from time import perf_counter

from .project import ROOT
from .records import fingerprint
from .job_workspace import changed_files, check_files
from .delegation_contract import SHARED_SYSTEM, worker_command, parse_worker


def select_assignments(events, cases):
    selected=[]
    for case in cases:
        declarations=[e for e in events if e.get("case_id")==case["id"] and e["type"]=="task_declared"]
        if declarations:
            task=declarations[0]["task"]
            selection="first actual declared task, regardless of its output"
            if task["depends_on"] or task["dependency_artifacts"]:
                raise ValueError("First-task selection unexpectedly requires prior artifacts")
        else:
            if case["id"]!="job02":
                raise ValueError("Only the preregistered no-delegation control may lack a task")
            task={"id":"no_delegation_control","command":"Review the existing configuration against the public requirements.",
                  "deliverable":"report or proposed edits","success_criteria":case["goal"],
                  "files":list(case["files"]),"depends_on":[],"dependency_artifacts":[],
                  "snapshot":{"version":0,"files":case["files"],"sha256":fingerprint(case["files"])}}
            selection="explicit no-delegation public-input control, not an actual Worker assignment"
        if fingerprint(task["snapshot"]["files"])!=task["snapshot"]["sha256"]:
            raise ValueError("Recorded snapshot identity changed")
        selected.append({"case":case,"task":task,"selection":selection})
    return selected


def run_execution_calibration(output_dir):
    import torch
    from .loading import load_models
    from .transcript import aligned_ids
    protocol=json.loads((ROOT/"configs/execution-calibration.json").read_text())
    path=ROOT/protocol["events_path"]
    if hashlib.sha256(path.read_bytes()).hexdigest()!=protocol["events_sha256"]:
        raise ValueError("Frozen source events changed")
    public=ROOT/"data/job-config-dev-eight/public.json"
    manifest=json.loads(public.with_name("manifest.json").read_text())
    if hashlib.sha256(public.read_bytes()).hexdigest()!=manifest["public_sha256"]:
        raise ValueError("Frozen public inputs changed")
    cases=json.loads(public.read_text())
    if [c["id"] for c in cases]!=protocol["case_ids"] or protocol["case_ids"]!=manifest["expected_case_ids"]:
        raise ValueError("Fixed eight-parent denominator changed")
    selections=select_assignments([json.loads(line) for line in path.read_text().splitlines()],cases)
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=False)
    (output/"selected-inputs.json").write_text(json.dumps(selections,ensure_ascii=False,indent=2),encoding="utf-8")
    config=json.loads((ROOT/"configs/cloud.json").read_text())
    lock=json.loads((ROOT/"configs/cloud.models.lock.json").read_text())
    result={"status":"running","protocol":protocol,"config":config,"model_pins":lock,
        "rows":[{"case_id":c["id"],"model":role,"status":"not_run"} for c in cases for role in protocol["model_order"]],
        "expected_pairs":8,"expected_model_cases":16,"training":False,"c2c_executed":False,
        "autonomous_actions_executed":False,"original_planning_kv_used":False,
        "source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*sorted((ROOT/"src/agent_runtime").glob("*.py")),ROOT/"scripts/run.py",ROOT/"configs/execution-calibration.json"]}}
    started=perf_counter()
    try:
        torch.set_num_threads(4);torch.manual_seed(config["seed"])
        load_started=perf_counter();combined,tokenizers=load_models(config,lock)
        result["loading_seconds"]=perf_counter()-load_started
        result["gpu"]=torch.cuda.get_device_name()
        index=0
        for selected in selections:
            case,task=selected["case"],selected["task"]
            # No reference answers, past assistant claims or old Worker reports are supplied.
            background={"goal":case["goal"],"project_notes":case["notes"],"public_rules":case["rules"],
                "missing_information":case.get("missing_information",[]),
                "authorized_files":{name:task["snapshot"]["files"][name] for name in task["files"]}}
            initial_ids=None
            for role in protocol["model_order"]:
                model=combined.model_list[1 if role=="sender" else 0];tokenizer=tokenizers[role]
                row={"case_id":case["id"],"model":role,"selection":selected["selection"],
                    "task":task,"calls":[],"tools":[],"status":"running","artifact":None,
                    "context_origin":"fresh native task execution context; no original Planner KV or mixed role history"}
                result["rows"][index]=row;index+=1
                row_started=perf_counter();observations=[]
                try:
                    for call in range(protocol["max_model_calls"]):
                        invocation=json.loads(worker_command(task,observations))
                        invocation["public_background"]=background
                        messages=[{"role":"system","content":SHARED_SYSTEM},
                                  {"role":"user","content":json.dumps(invocation,ensure_ascii=False)}]
                        text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
                        ids=aligned_ids(text,tokenizers["sender"],tokenizers["receiver"])
                        if call==0:
                            if initial_ids is None:initial_ids=ids
                            if initial_ids!=ids:raise ValueError("Initial paired input differs")
                        if len(ids)+protocol["max_new_tokens"]>config["max_input_tokens"]:
                            raise ValueError("Native input plus reserve exceeds frozen context budget")
                        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();began=perf_counter()
                        with torch.inference_mode():
                            generated=model.generate(input_ids=torch.tensor([ids],device=model.device),
                                attention_mask=torch.ones((1,len(ids)),device=model.device,dtype=torch.long),
                                do_sample=False,temperature=None,top_p=None,top_k=None,
                                repetition_penalty=1.0,no_repeat_ngram_size=0,
                                max_new_tokens=protocol["max_new_tokens"],pad_token_id=tokenizer.eos_token_id,use_cache=True)
                        torch.cuda.synchronize();seconds=perf_counter()-began
                        emitted=generated[0,len(ids):].tolist();del generated
                        eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
                        generation={"input_ids":ids,"input_sha256":fingerprint(ids),"messages":messages,
                            "generated_ids":emitted,"output_tokens":len(emitted),
                            "output":tokenizer.decode(emitted,skip_special_tokens=True),
                            "raw_output":tokenizer.decode(emitted,skip_special_tokens=False),
                            "stop_reason":"eos" if emitted and emitted[-1] in eos else "token_limit",
                            "output_constraints":False,"native_prefill_and_decode_seconds":seconds,
                            "peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20}
                        row["calls"].append(generation)
                        value=parse_worker(generation,task)
                        generation["validated_action"]=value
                        if value["action"] in {"report","need_info"}:
                            row.update(status="terminal_artifact",artifact=value)
                            break
                        tool_started=perf_counter()
                        if value["action"]=="inspect":
                            tool_result={"version":task["snapshot"]["version"],"files":background["authorized_files"]}
                        else:
                            tool_result=check_files(case,changed_files(task["snapshot"]["files"],value["changes"],task["files"]))
                        observation={"worker_action":value,"tool_result":tool_result}
                        observations.append(observation)
                        row["tools"].append({**observation,"seconds":perf_counter()-tool_started})
                    else:
                        row.update(status="model_call_budget_exhausted",error="No terminal artifact within the fixed two calls")
                except Exception as exc:
                    row.update(status="failed",error={"type":type(exc).__name__,"message":str(exc)})
                row["wall_seconds"]=perf_counter()-row_started
                (output/f"{case['id']}-{role}.json").write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding="utf-8")
                print(case["id"],role,row["status"],len(row["calls"]),round(row["wall_seconds"],3),flush=True)
        result["status"]="completed"
    except Exception as exc:
        result.update(status="failed",error={"type":type(exc).__name__,"message":str(exc)})
        raise
    finally:
        result["batch_wall_seconds"]=perf_counter()-started
        (output/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
