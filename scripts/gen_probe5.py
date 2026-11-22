"""生成 probe5 系列：4 个低步数 OS 场景（把多步计算压到 0.6B/冻结 4B 可及的一两步算术）。

probe4 的教训：FIFO 20 步重放、SCAN 8 请求求和、RR 全程模拟、令牌桶 9 请求
超出冻结 4B 工作笔记的能力，导致 analysis 为空、拒绝委派（4/4 确定性失败）。
probe5 保持同样的 OS 主题，但把计算压到 1-2 步，期望值全部经 Python 重模拟自验。
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "c2c-core-os"

OBS = ("All facts above are public observations. No hidden expected schedule is supplied. "
       "The controller is waiting on the pending decision value; a bounded verification of that "
       "value is the useful delegation now. Do not claim a blocked task ran.")


def case(cid, goal, runtime, tasks, decision, events, state, intended, expected, derivation):
    return {
        "id": f"os-c2c-probe5-{cid}",
        "goal": goal,
        "public_background": {
            "runtime": runtime,
            "tasks": tasks,
            "pending_decision": decision,
            "events": events,
        },
        "current_state": state,
        "probe": {
            "family": "derived_intermediate",
            "intended_question": intended,
            "expected_answer": expected,
            "derivation": derivation,
        },
    }


def task(tid, state, deps, cpu, mem, dur):
    return {"id": tid, "state": state, "depends_on": deps,
            "cpu": cpu, "gpu": 0, "memory_mb": mem, "duration_s": dur}


# ---------- 1. 页表翻译（十进制、单步乘法） ----------
PS = 4096
FRAME, OFF = 8, 1315
VA = FRAME * PS + OFF  # 34083
PA = FRAME * PS + OFF  # offset 即页内位移
assert VA == 34083 and PA == 34083
pagetable = case(
    "pagetable-01",
    "Bring the public address-translation state to a verified next step while preserving "
    "paging invariants and the fault audit trail.",
    {"page_size_bytes": PS,
     "translation_rule": "physical_address = frame_number * page_size + offset; the virtual "
                         "address 34083 lies in frame 8 at offset 1315 (8*4096=32768, 34083-32768=1315).",
     "pme": {"virtual_address": VA, "frame_number": FRAME, "offset": OFF, "present": True}},
    [task("walk", "READY", [], 1, 64, 1),
     task("tlb_fill", "BLOCKED", ["walk"], 1, 64, 1)],
    {"field": "physical_address_bytes", "question": "A present page map entry maps virtual frame 8 to physical "
     "frame 8 with page size 4096 B. For virtual address 34083 (offset 1315 within its page), "
     "what is the physical address in bytes?",
     "controller_note": "The memory manager is blocked until the translation is verified."},
    [{"fault_address": VA, "access": "read", "frame": FRAME, "offset": OFF}],
    {"page_faults": 1, "walks_completed": 0, "observation_boundary": OBS},
    "Physical address for virtual address 34083.",
    PA,
    f"frame 8 * 4096 = 32768; 32768 + offset 1315 = {PA}.",
)

# ---------- 2. FIFO 缺页（2 帧、5 步引用串） ----------
REFS5 = [1, 2, 3, 1, 2]
FRAMES5 = 2
frames, faults5, trace5 = [], 0, []
for r in REFS5:
    if r in frames:
        trace5.append(f"{r}: hit"); continue
    faults5 += 1
    ev = frames.pop(0) if len(frames) == FRAMES5 else None
    frames.append(r)
    trace5.append(f"{r}: fault (evict {ev})")
assert faults5 == 5, trace5
pagefifo = case(
    "pagefifo-01",
    "Bring the public page-fault ledger to a verified next step while preserving FIFO "
    "replacement invariants and the frame audit trail.",
    {"frames": FRAMES5, "replacement": "FIFO: on a fault with full frames, evict the oldest loaded page",
     "reference_string": REFS5, "page_size_bytes": 4096},
    [task("fault_replay", "READY", [], 1, 64, 2),
     task("thrash_report", "BLOCKED", ["fault_replay"], 1, 128, 2)],
    {"field": "fifo_page_faults", "question": "With 2 frames and FIFO replacement, how many page faults occur "
     "for the reference string 1,2,3,1,2?",
     "controller_note": "The memory manager is blocked until the fault count is verified."},
    [{"time_step": i, "page": r} for i, r in enumerate(REFS5)],
    {"frames_content": [], "faults_so_far": 0, "hits_so_far": 0, "observation_boundary": OBS},
    "FIFO page fault count for the 5-step reference string with 2 frames.",
    faults5,
    "; ".join(trace5) + f"; all 5 references fault (2 frames cannot hold the working set), total = {faults5}.",
)

# ---------- 3. 令牌桶（4 请求、小数字） ----------
CAP5, RATE5 = 10, 1
ARR5, SIZE5 = [0, 1, 2, 3], [3, 2, 4, 5]
tokens, last, adm5, tr5 = CAP5, 0, 0, []
for t, s in zip(ARR5, SIZE5):
    tokens = min(CAP5, tokens + RATE5 * (t - last)); last = t
    if tokens >= s:
        tokens -= s; adm5 += 1; tr5.append(f"t={t} size={s}: admit (left {tokens})")
    else:
        tr5.append(f"t={t} size={s}: DROP (only {tokens})")
assert adm5 == 3, tr5
ratelimit = case(
    "ratelimit-01",
    "Bring the public gateway state to a compliant next step while preserving token bucket "
    "invariants and the admission audit trail.",
    {"bucket_capacity_tokens": CAP5, "refill_tokens_per_second": RATE5,
     "algorithm": "Discrete token bucket: on each request at time t, tokens = min(capacity, tokens + rate*(t-last_t)); "
                  "admit iff tokens >= size, consuming s tokens, else drop.",
     "audit_window_s": 3},
    [task("meter", "SUCCEEDED", [], 1, 256, 3),
     task("ledger_flush", "READY", ["meter"], 1, 512, 2)],
    {"field": "admitted_requests", "question": "How many of the 4 requests in the audit window are admitted by "
     "the token bucket (capacity 10, refill 1 token/s, requests at t=0,1,2,3 with sizes 3,2,4,5)?",
     "controller_note": "The rate-limit controller is blocked until the admission count is verified."},
    [{"request_time_s": t, "request_size_tokens": s} for t, s in zip(ARR5, SIZE5)],
    {"bucket_tokens_now": CAP5, "last_refill_time_s": 0, "requests_seen": 4, "drops_seen": 0,
     "observation_boundary": OBS},
    "Token bucket admission count over the 4-request audit window.",
    adm5,
    "; ".join(tr5) + f"; admitted = {adm5} of 4.",
)

# ---------- 4. SCAN（4 请求、两步求和） ----------
HEAD5, QUEUE5 = 53, [65, 98, 14, 37]
up5 = sorted(x for x in QUEUE5 if x >= HEAD5)
down5 = sorted((x for x in QUEUE5 if x < HEAD5), reverse=True)
seq5 = up5 + down5
moves5 = [abs(seq5[0] - HEAD5)] + [abs(b - a) for a, b in zip(seq5, seq5[1:])]
tot5 = sum(moves5)
assert seq5 == [65, 98, 37, 14] and tot5 == 129, (seq5, tot5)
disk = case(
    "diskscan-01",
    "Bring the public I/O scheduler state to a verified next step while preserving elevator "
    "policy and the head-movement audit trail.",
    {"cylinders": 200, "head_start": HEAD5, "direction": "up",
     "algorithm": "SCAN (elevator): serve pending requests while moving in the current direction; reverse "
                  "only after serving the last request in that direction.",
     "queue_at_t0": QUEUE5},
    [task("queue_snapshot", "SUCCEEDED", [], 1, 64, 1),
     task("seek_audit", "READY", ["queue_snapshot"], 1, 64, 1)],
    {"field": "total_head_movement", "question": "The head is at cylinder 53 moving toward higher cylinders; "
     "the pending queue is 65, 98, 14, 37. Under SCAN, what is the total head movement in cylinders "
     "to serve the whole queue?",
     "controller_note": "The I/O controller is blocked until the seek total is verified."},
    [{"head_at_cylinder": HEAD5, "moving": "up"}, {"queue_snapshot": QUEUE5}],
    {"requests_pending": 4, "head_position": HEAD5, "serviced": 0, "observation_boundary": OBS},
    "Total SCAN head movement for the 4-request queue.",
    tot5,
    f"service order {seq5}; movement 12 + 33 (up) + 61 + 23 (down) = {tot5} cylinders.",
)

CASES = [pagetable, pagefifo, ratelimit, disk]
for c in CASES:
    name = c["id"].replace("os-c2c-probe5-", "")
    path = OUT / f"public-probe5-{name}.json"
    path.write_text(json.dumps(c, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{path.name}: expected={c['probe']['expected_answer']}")
print("done,", len(CASES), "cases")
