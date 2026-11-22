"""Versioned review deliverables. Structural readiness never proves semantic truth.

This domain boundary performs no planning, generation, retrieval ranking or grading.
It is not yet connected to the autonomous model scheduler.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from uuid import uuid4

from .records import fingerprint


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _keys(value, expected):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("Unexpected review fields")


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON field")
        value[key] = item
    return value


class ReviewPacket:
    """Loads only an explicit public packet and hash-locked source files."""

    def __init__(self, directory):
        self.root = Path(directory).resolve()
        self.public = json.loads((self.root / "public.json").read_text(encoding="utf-8"))
        self.lock = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if hashlib.sha256((self.root / "public.json").read_bytes()).hexdigest() != self.lock["public_sha256"]:
            raise ValueError("Public packet identity changed")
        self.items = {item["id"]: item for item in self.public["items"]}
        if not self.items or len(self.items) != len(self.public["items"]):
            raise ValueError("Expected unique review items")
        self.sources = {}
        for source in self.public["sources"]:
            path = (self.root / source["path"]).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise ValueError("Source outside the public packet")
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != source["sha256"]:
                raise ValueError("Source identity changed")
            if source["id"] in self.sources:
                raise ValueError("Duplicate source ID")
            self.sources[source["id"]] = {**source, "text": raw.decode("utf-8"),
                                          "lines": raw.decode("utf-8").splitlines(keepends=True)}
        self.identity = fingerprint({"public": self.public, "source_hashes":
                                     {sid: s["sha256"] for sid, s in self.sources.items()}})

    def quote(self, source_id, first_line, last_line):
        """Exact 1-based source lookup; no fuzzy repair or semantic search."""
        if source_id not in self.sources:
            raise ValueError("Unknown source")
        source = self.sources[source_id]
        if (type(first_line) is not int or type(last_line) is not int or
                not 1 <= first_line <= last_line <= len(source["lines"])):
            raise ValueError("Invalid exact line range")
        return {"source_id": source_id, "source_sha256": source["sha256"],
                "first_line": first_line, "last_line": last_line,
                "quote": "".join(source["lines"][first_line - 1:last_line])}

    def validate_findings(self, findings, assigned_items):
        if not isinstance(findings, list) or not findings:
            raise ValueError("A report needs actual findings")
        allowed = set(assigned_items)
        if not allowed or not allowed <= set(self.items):
            raise ValueError("Unknown or empty assignment scope")
        seen = set()
        for finding in findings:
            _keys(finding, {"item_id", "disposition", "conclusion", "evidence", "missing_facts"})
            item = finding["item_id"]
            if not isinstance(item, str) or item not in allowed or item in seen:
                raise ValueError("Duplicate or unauthorized finding")
            seen.add(item)
            if finding["disposition"] not in {"impact", "no_impact", "needs_information"}:
                raise ValueError("Unknown disposition")
            if not _text(finding["conclusion"]):
                raise ValueError("Missing actual conclusion")
            missing = finding["missing_facts"]
            if not isinstance(missing, list) or any(not _text(fact) for fact in missing):
                raise ValueError("Missing facts must be explicit text")
            if finding["disposition"] == "needs_information" and not missing:
                raise ValueError("A missing-information disposition needs explicit missing facts")
            # Known upstream impact can coexist with missing project-specific facts.
            citations = finding["evidence"]
            if not isinstance(citations, list):
                raise ValueError("Evidence must be a list")
            if not citations and finding["disposition"] != "needs_information":
                raise ValueError("A conclusive finding requires exact evidence")
            for citation in citations:
                _keys(citation, {"source_id", "source_sha256", "first_line", "last_line", "quote"})
                expected = self.quote(citation["source_id"], citation["first_line"], citation["last_line"])
                if citation != expected:
                    raise ValueError("Citation differs from the pinned original")
        # Partial reports remain partial: the host does not supply missing answers.
        return {"artifact_integrity_valid": True, "reported_items": sorted(seen),
                "unreported_assigned_items": sorted(allowed - seen), "semantic_correctness": "unassessed"}


class ReviewLedger:
    """Append-only raw receipts and versioned Planner selections.

    Caller supplies authentic actor/task/input provenance. Fixture provenance is
    supported but cannot be promoted to model provenance by this module.
    """

    def __init__(self, packet, directory):
        self.packet, self.root = packet, Path(directory)
        self.root.mkdir(parents=True, exist_ok=False)
        self.version = 0
        self.selected = {}
        self.receipts = {}
        self._save_revision()

    def _save_revision(self):
        value = {"version": self.version, "packet_sha256": self.packet.identity,
                 "selected": deepcopy(self.selected), "semantic_correctness": "unassessed"}
        value["sha256"] = fingerprint(value)
        with (self.root / f"revision-{self.version:04d}.json").open("x", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, indent=2)
        self.revision_hash = value["sha256"]

    def receive(self, raw_output, *, producer, task_id, assigned_items, provenance, stop_reason):
        if not _text(producer) or not _text(task_id) or not isinstance(raw_output, str):
            raise ValueError("Missing authentic producer, task or raw output")
        if not isinstance(provenance, dict) or provenance.get("origin") not in {"manual_contract_fixture", "model_generation"}:
            raise ValueError("Explicit provenance required")
        if provenance["origin"] == "model_generation":
            for key in ("request_id", "context_version", "input_sha256", "generation_record_sha256"):
                if not _text(provenance.get(key)):
                    raise ValueError("Missing model generation provenance")
        receipt = {"id": uuid4().hex, "packet_sha256": self.packet.identity, "producer": producer,
                   "task_id": task_id, "assigned_items": list(assigned_items),
                   "provenance": deepcopy(provenance), "raw_output": raw_output,
                   "stop_reason": stop_reason, "semantic_correctness": "unassessed"}
        try:
            if stop_reason != "eos":
                raise ValueError("Incomplete generation cannot deliver a report")
            text = raw_output.strip()
            envelope = re.fullmatch(r"```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```", text)
            receipt["outer_fence_removed"] = envelope is not None
            value = json.loads(envelope.group(1) if envelope else text, object_pairs_hook=_unique_object)
            _keys(value, {"action", "findings"})
            if value["action"] != "report":
                raise ValueError("Expected an actual report action")
            receipt["validation"] = self.packet.validate_findings(value["findings"], assigned_items)
            receipt["value"] = value
        except (ValueError, TypeError, KeyError) as exc:
            receipt["validation"] = {"artifact_integrity_valid": False, "error": str(exc),
                                     "semantic_correctness": "unassessed"}
        receipt["sha256"] = fingerprint(receipt)
        with (self.root / f"receipt-{receipt['id']}.json").open("x", encoding="utf-8") as file:
            json.dump(receipt, file, ensure_ascii=False, indent=2)
        self.receipts[receipt["sha256"]] = deepcopy(receipt)
        return deepcopy(receipt)

    def publish(self, selections, *, base_version, base_sha256, producer):
        """Select exact received findings; editing requires a NEW attributed receipt."""
        if producer != "Planner":
            raise ValueError("Only Planner may publish the integrated review")
        if type(base_version) is not int or not isinstance(base_sha256, str):
            raise ValueError("Expected an exact integer version and revision hash")
        if base_version != self.version or base_sha256 != self.revision_hash:
            return {"published": False, "error": "version_conflict", "current_version": self.version}
        if not isinstance(selections, dict) or not selections or not set(selections) <= set(self.packet.items):
            raise ValueError("Expected explicit item-to-receipt selections")
        updates = {}
        for item, sha in selections.items():
            receipt = self.receipts.get(sha)
            if not receipt or not receipt["validation"]["artifact_integrity_valid"]:
                raise ValueError("Unknown or invalid deliverable")
            path = self.root / f"receipt-{receipt['id']}.json"
            actual = json.loads(path.read_text(encoding="utf-8"))
            if actual != receipt or fingerprint({k: v for k, v in actual.items() if k != "sha256"}) != sha:
                raise ValueError("Receipt changed after delivery")
            findings = [f for f in receipt["value"]["findings"] if f["item_id"] == item]
            if len(findings) != 1:
                raise ValueError("Selected receipt does not report the requested item")
            updates[item] = {"receipt_sha256": sha, "producer": receipt["producer"],
                             "task_id": receipt["task_id"], "finding": deepcopy(findings[0]),
                             "provenance": deepcopy(receipt["provenance"])}
        self.selected.update(updates)
        self.version += 1
        self._save_revision()
        return {"published": True, **self.readiness()}

    def readiness(self):
        return {"version": self.version, "revision_sha256": self.revision_hash,
                "coverage_complete": set(self.selected) == set(self.packet.items),
                "unreported_items": sorted(set(self.packet.items) - set(self.selected)),
                "unresolved_items": sorted(item for item, record in self.selected.items()
                                           if record["finding"]["disposition"] == "needs_information"
                                           or record["finding"]["missing_facts"]),
                "semantic_correctness": "unassessed", "task_success": None,
                "scope": "Coverage and evidence identity only; no semantic completion oracle"}
