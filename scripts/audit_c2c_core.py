"""Offline semantic audit for one saved OS C2C core result.

This audit reads only the public fixture, its local offline acceptance facts,
and raw Worker outputs.  It never selects a case from model results and does
not treat JSON validity, EOS, cache hashes, or fusion as semantic success.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_report(text):
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Worker output is not an object")
    report = value.get("report")
    if report is None:
        report = value.get("action_report")
    if isinstance(report, dict):
        return json.dumps(report, ensure_ascii=False).lower()
    if isinstance(report, str):
        return report.lower()
    return json.dumps(value, ensure_ascii=False).lower()


def _audit_arm(arm, required, profile="core"):
    output = arm.get("output", "")
    checks = {"json_object": False, "required_facts": {}, "contradiction_free": False}
    try:
        text = _raw_report(output)
        checks["json_object"] = True
    except Exception as exc:
        return {"semantic_success": False, "error": f"{type(exc).__name__}: {exc}", "checks": checks}
    if profile == "rich_planning":
        encode_positive = any(
            "encode" in sentence and (
                "succeeded" in sentence or "completed successfully" in sentence
            ) and "not " not in sentence and "whether" not in sentence and "if " not in sentence
            for sentence in re.split(r"[.!?]", text)
        )
        checks["required_facts"] = {
            "encode_not_terminal_success": (
                "encode" in text and any(mark in text for mark in
                ("running", "not completed successfully", "not terminal-success", "not in a terminal-success"))
                and not encode_positive),
            "rank_status_present": "rank" in text,
        }
    else:
        checks["required_facts"] = {
            "encode_not_terminal_success": (
                "encode" in text and "terminal-success" in text
                and any(mark in text for mark in ("not terminal-success", "not in a terminal-success", "not yet terminal-success"))),
            "rank_blocked_by_encode": "rank" in text and "blocked" in text and "encode" in text,
        }
    sentences = [part.strip() for part in re.split(r"[.!?]", text)]
    positive_encode = any(
        "encode" in sentence and "completed successfully" in sentence
        and "not " not in sentence and "whether" not in sentence and "if " not in sentence
        for sentence in sentences
    )
    positive_rank = any(
        "rank" in sentence and re.search(r"(?:can|may) be scheduled", sentence)
        for sentence in sentences
    )
    checks["contradiction_free"] = not (positive_encode or positive_rank)
    checks["required_facts"]["offline_required_facts_loaded"] = bool(required)
    expected_keys = ("encode_not_terminal_success", "rank_status_present") if profile == "rich_planning" else ("encode_not_terminal_success", "rank_blocked_by_encode")
    success = checks["json_object"] and all(checks["required_facts"][key] for key in expected_keys) and checks["contradiction_free"]
    return {"semantic_success": success, "checks": checks, "raw_output": output}


def audit(result_path, expected_path, output_path):
    result_path = Path(result_path)
    expected_path = Path(expected_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    if result.get("case_id") != expected.get("case_id"):
        raise ValueError("Result and offline expected case IDs differ")
    required = expected.get("required_facts", [])
    profile = expected.get("profile", "core")
    tasks = (result.get("planner") or {}).get("action", {}).get("tasks") or []
    public_path = expected_path.with_name("public.json")
    public = json.loads(public_path.read_text(encoding="utf-8"))
    states = {task["id"]: task["state"] for task in public["public_background"]["tasks"]}
    dependency_contract_valid = all(
        states.get(dep) == "SUCCEEDED"
        for task in tasks for dep in task.get("depends_on", [])
    )
    commands = [arm.get("command") for arm in result.get("arms", {}).values() if isinstance(arm, dict)]
    audit_result = {
        "status": "completed",
        "result_path": str(result_path),
        "result_sha256": _sha(result_path),
        "expected_path": str(expected_path),
        "expected_sha256": _sha(expected_path),
        "case_id": expected["case_id"],
        "audit_policy": expected.get("audit_policy"),
        "planner_checks": {
            "actual_delegate": (result.get("planner") or {}).get("action", {}).get("action") == "delegate",
            "actual_task_count": len(tasks),
            "task_command": tasks[0].get("command") if tasks else None,
            "all_worker_commands_identical": len(set(commands)) <= 1,
            "dependency_contract_valid": dependency_contract_valid,
            "dependency_contract_note": (
                "Every declared dependency is terminal-success in the public state"
                if dependency_contract_valid else
                "Planner declared a dependency that is not terminal-success; preserved as a Planner contract failure"
            ),
        },
        "arms": {},
        "c2c_checks": {
            "worker_consumed_fused_cache": bool((result.get("arms", {}).get("c2c_shared") or {}).get("worker_consumed_fused_cache")),
            "source_cache_was_reencoded": (result.get("arms", {}).get("c2c_shared") or {}).get("source_cache_was_reencoded"),
            "fusion_seconds_positive": (result.get("arms", {}).get("c2c_shared") or {}).get("fusion_seconds", 0) > 0,
            "physical_zero_copy": (result.get("preparation") or {}).get("physical_zero_copy"),
        },
        "costs": {
            "planner_handoff_generation_seconds": (result.get("preparation") or {}).get("text_brief_extra_generation_seconds"),
            "t2t_receiver_prefill_seconds": (result.get("preparation") or {}).get("text_brief_receiver_prefill_seconds"),
            "c2c_fusion_seconds": (result.get("preparation") or {}).get("c2c_fusion_seconds"),
            "kv_bytes_are_not_network_traffic": (result.get("preparation") or {}).get("kv_bytes_are_not_network_traffic"),
        },
        "feedback_checks": {},
    }
    for name, row in (result.get("feedback_planner") or {}).items():
        observation = row.get("observation") or {}
        raw = observation.get("worker_output", "")
        required_fields = (
            "worker_output_sha256", "public_state_sha256", "arm_method",
            "source_cache_sha256", "receiver_cache_sha256", "context_version",
        )
        audit_result["feedback_checks"][name] = {
            "provenance_fields_present": all(field in observation for field in required_fields),
            "worker_output_sha256_matches": (
                hashlib.sha256(raw.encode("utf-8")).hexdigest()
                == observation.get("worker_output_sha256")
            ),
            "consumed_worker_observation": row.get("consumed_worker_observation"),
            "consumption_evidence": row.get("consumption_evidence"),
            "parent_acceptance_gate": row.get("parent_acceptance_gate"),
            "feedback_analysis_cache_recorded": bool(
                (row.get("analysis") or {}).get("source_cache_sha256")
            ),
        }
    for name, arm in result.get("arms", {}).items():
        audit_result["arms"][name] = _audit_arm(arm, required, profile)
    audit_result["semantic_summary"] = {
        name: row.get("semantic_success") for name, row in audit_result["arms"].items()
    }
    audit_result["c2c_passes_semantic_parity_with_text_full"] = (
        audit_result["semantic_summary"].get("c2c_shared") is True
        and audit_result["semantic_summary"].get("text_full") is True
    )
    output_path = Path(output_path)
    output_path.write_text(json.dumps(audit_result, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--expected", type=Path,
                        default=Path("data/c2c-core-os/offline_expected.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.result, args.expected, args.output)


if __name__ == "__main__":
    main()
