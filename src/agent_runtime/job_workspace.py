"""Versioned task files and public rule checks, never a repair solver."""
from copy import deepcopy
import json
from pathlib import Path
from .records import fingerprint

FIELDS = ("cpu", "memory_mb", "parallelism", "retries", "timeout_s")


def check_files(case, files):
    violations, missing = [], list(case.get("missing_information", []))
    original = case["files"]
    if set(files) != set(original):
        violations.append({"rule": "preserve_file_set"})
    for name, initial in original.items():
        value = files.get(name, {})
        if set(value) != set(initial):
            violations.append({"file": name, "rule": "preserve_fields"})
        for field in set(initial)-set(FIELDS):
            if value.get(field) != initial[field]:
                violations.append({"file": name, "field": field, "rule": "preserve_immutable_value"})
        for field in FIELDS:
            if type(value.get(field)) is not int or value[field] < 0:
                violations.append({"file": name, "field": field, "rule": "nonnegative_integer"})
    if violations:
        return {"passed": False, "violations": violations, "missing_information": missing}
    for rule in case["rules"]:
        kind = rule["kind"]
        if kind == "range":
            actual = files[rule["file"]][rule["field"]]
            passed = rule["minimum"] <= actual <= rule["maximum"]
        elif kind == "equal":
            actual = files[rule["file"]][rule["field"]]
            passed = actual == rule["value"]
        elif kind == "field_le":
            actual = {f: files[rule["file"]][f] for f in (rule["left"],rule["right"])}
            passed = actual[rule["left"]] <= actual[rule["right"]]
        elif kind == "sum_le":
            actual = sum(files[f][rule["field"]] for f in rule["files"])
            passed = actual <= rule["maximum"]
        else:
            raise ValueError("Unknown public rule: " + kind)
        if not passed:
            violations.append({"rule": rule, "observed": actual})
    return {"passed": not violations and not missing,
            "violations": violations, "missing_information": missing}


def changed_files(files, changes, allowed_files):
    if not isinstance(changes, list) or len(changes) > 20:
        raise ValueError("Expected at most 20 explicit field edits")
    result, seen = deepcopy(files), set()
    for edit in changes:
        if not isinstance(edit, dict) or set(edit) != {"file", "field", "value"}:
            raise ValueError("Each edit needs exactly file, field, value")
        name, field, value = edit["file"], edit["field"], edit["value"]
        if name not in allowed_files or field not in FIELDS:
            raise ValueError("Edit outside granted files or mutable fields")
        if type(value) is not int or value < 0 or (name,field) in seen:
            raise ValueError("Expected unique edits with nonnegative integer values")
        seen.add((name,field))
        result[name][field] = value
    return result


def proposal_effect(case, original, candidate):
    """Compare public checks on one snapshot; neither solve nor rank proposals."""
    before, after = check_files(case, original), check_files(case, candidate)
    structural = any(not isinstance(v["rule"], dict)
                     for check in (before, after) for v in check["violations"])
    result = {"base_files_sha256": fingerprint(original),
              "candidate_files_sha256": fingerprint(candidate),
              "files_changed": original != candidate,
              "base_parent_check": before, "candidate_parent_check": after,
              "rule_comparison_available": not structural,
              "resolved_violations": None, "persisting_violations": None,
              "introduced_violations": None,
              "scope": "public rule effects on original snapshot; not subtask success, utility score or acceptance advice"}
    if structural:
        result["comparison_error"] = "Structural validation failed; numeric rules may not have run"
        return result
    # Identity excludes observed values. Preserve duplicate rules as a multiset.
    remaining = {}
    for violation in after["violations"]:
        key = fingerprint({k:v for k,v in violation.items() if k != "observed"})
        remaining.setdefault(key, []).append(violation)
    resolved, persisting = [], []
    for violation in before["violations"]:
        key = fingerprint({k:v for k,v in violation.items() if k != "observed"})
        matches = remaining.get(key, [])
        if matches:
            persisting.append({"before": violation, "after": matches.pop(0)})
        else:
            resolved.append(violation)
    result.update(resolved_violations=resolved, persisting_violations=persisting,
                  introduced_violations=[v for group in remaining.values() for v in group])
    return result


def proposal_snapshot_status(snapshot, current, artifact=None):
    """Freshness is independent from whether the original proposal was useful."""
    if fingerprint(snapshot["files"]) != snapshot["sha256"]:
        raise ValueError("Proposal base snapshot content/hash mismatch")
    if fingerprint(current["files"]) != current["sha256"]:
        raise ValueError("Current snapshot content/hash mismatch")
    if artifact is not None and (artifact["base_version"] != snapshot["version"] or
                                 artifact["base_sha256"] != snapshot["sha256"]):
        raise ValueError("Artifact is not bound to its recorded dispatch snapshot")
    version_matches = snapshot["version"] == current["version"]
    hash_matches = snapshot["sha256"] == current["sha256"]
    return {"base_version": snapshot["version"], "current_version": current["version"],
            "base_sha256": snapshot["sha256"], "current_sha256": current["sha256"],
            "version_matches": version_matches, "hash_matches": hash_matches,
            "snapshot_matches_current": version_matches and hash_matches,
            "commit_revalidates_snapshot": True}


def proposal_effect_observation(effect):
    """One rule description per transition; identities live in the enclosing artifact."""
    observation = {k:v for k,v in effect.items()
                   if k not in {"base_parent_check", "candidate_parent_check", "persisting_violations",
                                "base_files_sha256", "candidate_files_sha256", "scope"}}
    observation["base_check_passed"] = effect["base_parent_check"]["passed"]
    observation["scope"] = "public rules on original snapshot; not acceptance advice"
    observation["persisting_violations"] = None if effect["persisting_violations"] is None else [
        {"rule": pair["before"]["rule"], "before": pair["before"].get("observed"),
         "after": pair["after"].get("observed")}
        for pair in effect["persisting_violations"]]
    return observation


def proposal_check_observation(effect):
    """Numerical rule details are already in the effect; retain structural errors."""
    after = effect["candidate_parent_check"]
    result = {"passed": after["passed"], "missing_information": after["missing_information"]}
    if not effect["rule_comparison_available"]:
        result["violations"] = after["violations"]
    return result


def proposal_freshness_observation(snapshot, current, artifact=None):
    status = proposal_snapshot_status(snapshot, current, artifact)
    # Exact base and current identities are already present in the enclosing state.
    return {k:status[k] for k in ("version_matches", "hash_matches", "snapshot_matches_current")}


class JobWorkspace:
    def __init__(self, case, directory):
        self.case, self.root = case, Path(directory)
        self.root.mkdir(parents=True, exist_ok=False)
        self.version, self.files = 0, deepcopy(case["files"])
        for name in self.files:
            if Path(name).name != name or not name.endswith(".json"):
                raise ValueError("Task file names must be simple JSON basenames")
        self._persist("initial", [])

    def snapshot(self):
        return {"version": self.version, "files": deepcopy(self.files),
                "file_hashes": {n:fingerprint(v) for n,v in self.files.items()},
                "sha256": fingerprint(self.files)}

    def _persist(self, producer, changes):
        version_dir = self.root / f"v{self.version}"
        version_dir.mkdir(exist_ok=False)
        for name, value in self.files.items():
            (version_dir/name).write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")
        (version_dir/"manifest.json").write_text(json.dumps({**self.snapshot(),
            "producer":producer,"changes":changes}, indent=2)+"\n", encoding="utf-8")

    def commit(self, base_version, base_hash, changes, producer, allowed_files=None):
        if base_version != self.version or base_hash != fingerprint(self.files):
            return {"committed":False,"error":"version_conflict", "current_version":self.version,
                    "expected_base_version":base_version}
        updated = changed_files(self.files, changes, list(self.files) if allowed_files is None else allowed_files)
        before = self.snapshot()
        if not changes:
            return {"committed":False,"error":"empty_edit_list","current_version":self.version}
        if updated == self.files:
            # Acknowledging a proposal must not invalidate other snapshots when
            # no file changed. This says nothing about task correctness.
            return {"committed":False,"unchanged":True,"files_changed":False,
                    "producer":producer,"version":self.version,"sha256":before["sha256"],
                    "base_sha256":before["sha256"],"changes":changes,"semantic_verified":False}
        self.files, self.version = updated, self.version+1
        self._persist(producer, changes)
        return {"committed":True,"producer":producer,"base_sha256":before["sha256"],
                "version":self.version,"sha256":fingerprint(self.files),"changes":changes,
                "files_changed":self.files!=before["files"],
                "semantic_verified":False}

    def check(self):
        # Read the delivered files, not just the in-memory candidate.
        actual = {n:json.loads((self.root/f"v{self.version}"/n).read_text()) for n in self.files}
        if actual != self.files:
            raise RuntimeError("Delivered files differ from versioned state")
        return {"version":self.version, "sha256":fingerprint(actual), **check_files(self.case, actual)}
