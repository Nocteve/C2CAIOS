"""Explicit family/parent/source-identity exclusion for prepared research data.

This checks declared provenance and exact identities, not semantic near-duplicate
detection. Family assignment must be reviewed before labeling or model runs.
"""
import json
from pathlib import Path


def validate_partitions(records, registry):
    excluded = {key: set(registry[key]) for key in
                ("excluded_parent_ids", "excluded_family_ids", "excluded_source_sha256")}
    seen = {"parent": {}, "family": {}, "source": {}}
    # Previously prepared candidates keep their split even when only a new
    # subset is validated. Re-declaring the same lineage in its split is fine.
    for reserved in registry.get("reserved_lineage_records", []):
        for kind,key in (("parent","parent_ids"),("family","family_ids"),("source","source_sha256")):
            for value in reserved[key]:
                previous=seen[kind].get(value)
                if previous is not None and previous!=reserved["split"]:
                    raise ValueError("Registry contains conflicting reserved splits")
                seen[kind][value]=reserved["split"]
    failures = []
    allowed_splits = {"train", "validation", "formal_test", "contract_dev_excluded"}
    ids = set()
    for record in records:
        rid = record.get("id")
        if not isinstance(rid, str) or not rid or rid in ids:
            failures.append({"id": rid, "error": "missing_or_duplicate_record_id"})
        ids.add(rid) if isinstance(rid, str) else None
        split = record.get("split")
        keys = {"parent": record.get("parent_ids"), "family": record.get("family_ids"),
                "source": record.get("source_sha256")}
        if split not in allowed_splits or any(not isinstance(v, list) or not v or
                any(not isinstance(x, str) or not x for x in v) for v in keys.values()):
            failures.append({"id": rid, "error": "incomplete_split_or_lineage"})
            continue
        if split != "contract_dev_excluded":
            for kind, excluded_key in (("parent", "excluded_parent_ids"),
                                       ("family", "excluded_family_ids"),
                                       ("source", "excluded_source_sha256")):
                overlaps = sorted(set(keys[kind]) & excluded[excluded_key])
                if overlaps:
                    failures.append({"id": rid, "error": "frozen_development_lineage", "kind": kind, "values": overlaps})
        for kind, values in keys.items():
            for value in values:
                previous = seen[kind].get(value)
                if previous is not None and previous != split:
                    failures.append({"id": rid, "error": "cross_split_lineage", "kind": kind,
                                     "value": value, "splits": [previous, split]})
                seen[kind][value] = split
    return {"partition_valid": not failures, "records": len(records), "failures": failures,
            "training_ready": False,
            "training_readiness_scope": "This is only a partition gate; it never certifies targets, semantic quality or training readiness.",
            "limitation": "Declared lineage and exact hashes only; mislabeled families and semantic paraphrases require separate review."}


def load_registry(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
