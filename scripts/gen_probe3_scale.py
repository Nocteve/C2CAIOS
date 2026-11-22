"""Background-scale variants of selected probe3 scenarios (2026-10-17).

Purpose: isolate how each communication arm behaves as the shared public
background grows (PPT hypothesis: expensive background reused by many Workers).
Each base scenario gets -b4 and -b16 variants whose public_background carries
an "archived_tasks" list of clearly-irrelevant historical jobs (terminal or
running maintenance records, marked as unrelated to the pending decision).
The probe question, expected answer and all decision-relevant fields are
byte-identical to the base scenario; only background volume changes.

Leakage rule (same as gen_probe3): the expected answer must not appear as a
standalone integer anywhere in the public JSON, including archived_tasks.
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_probe3 import SCENARIOS, GOAL, standalone_integers  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "c2c-core-os"

BASE_IDS = [
    "os-c2c-probe3-admission-mem-02",
    "os-c2c-probe3-critical-path-01",
    "os-c2c-probe3-latest-start-02",
    "os-c2c-probe3-gpu-waves-02",
    "os-c2c-probe3-transfer-02",
    "os-c2c-probe3-queue-wait-02",
]

TARGET_TOKENS = {"b4": 4000, "b10": 10000}
CHARS_PER_TOKEN = 3.4

ARCHIVE_NOTE = ("archived_tasks are historical records only: they are already "
                "terminal or routine maintenance and are NOT part of the pending "
                "decision; do not include them in any calculation for it.")

STATES = ["SUCCEEDED"] * 7 + ["BLOCKED"] * 2 + ["RUNNING"]
VERB = ["reindex", "compact", "snapshot", "audit", "garbage_collect", "checksum",
        "rotate", "prune", "dedupe", "export", "verify", "defrag"]
NOUN = ["shard", "volume", "table", "segment", "cache", "log", "image", "index"]


def filler_tasks(rng, count, forbidden):
    tasks = []
    for i in range(count):
        # id numbers start at 1000: a zero-padded numeric id parses to its
        # integer, which must never equal a forbidden expected answer.
        for _attempt in range(50):
            t = {
                "id": f"hist_{i + 1000:04d}",
                "state": rng.choice(STATES),
                "depends_on": [],
                "cpu": rng.randint(1, 4),
                "gpu": 0,
                "memory_mb": rng.choice([512, 1024, 1536, 2560, 3072]),
                "duration_s": rng.randint(20, 600),
            }
            if not (standalone_integers(t) & forbidden):
                break
        else:
            raise AssertionError("cannot generate non-leaking filler")
        tasks.append(t)
    return tasks


def estimate_tokens(case):
    return len(json.dumps(case, ensure_ascii=False)) / CHARS_PER_TOKEN


def main():
    rng = random.Random(20261017)
    written = 0
    for spec in SCENARIOS:
        if spec["id"] not in BASE_IDS:
            continue
        slug = spec["id"].split("probe3-")[1]
        base_case = json.loads((OUT / spec["file"]).read_text(encoding="utf-8"))
        forbidden = {base_case["probe"]["expected_answer"]}
        base_tokens = estimate_tokens(base_case)
        for tag, target in TARGET_TOKENS.items():
            case = json.loads(json.dumps(base_case))  # deep copy
            # size the filler list from the measured base volume
            per_filler = estimate_tokens({"archived_tasks": [filler_tasks(rng, 1, forbidden)[0]]})
            count = max(1, int((target - base_tokens) / max(per_filler, 1)))
            archived = filler_tasks(rng, count, forbidden)
            case["public_background"]["archived_tasks"] = archived
            note = case["current_state"].get("observation_boundary", "")
            if ARCHIVE_NOTE not in note:
                case["current_state"]["observation_boundary"] = note + " " + ARCHIVE_NOTE
            public_ints = standalone_integers(
                {"goal": case["goal"], "public_background": case["public_background"],
                 "current_state": case["current_state"]})
            answer = case["probe"]["expected_answer"]
            assert answer not in public_ints, f"{case['id']}: answer leaks in {tag}"
            assert len(public_ints) >= 4
            fname = f"public-probe3s-{slug}-{tag}.json"
            (OUT / fname).write_text(json.dumps(case, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")
            print(f"wrote {fname}: ~{estimate_tokens(case):.0f} est tokens, "
                  f"{len(archived)} archived tasks, answer={answer}")
            written += 1
    print(f"total {written} scale variants")


if __name__ == "__main__":
    main()
