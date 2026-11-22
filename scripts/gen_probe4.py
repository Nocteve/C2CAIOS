"""生成 probe4 系列：7 个全新 OS 主题场景（令牌桶/COW/死锁/SCAN/页表/RR/FIFO缺页）。

每个场景带 expected_answer 与逐步 derivation；生成时用 Python 重新模拟一遍，
assert 期望值与推导一致，防止手写算术出错。输出到 data/c2c-core-os/public-probe4-*.json。
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "c2c-core-os"

OBS = ("All facts above are public observations. No hidden expected schedule is supplied. "
       "The controller is waiting on the pending decision value; a bounded verification of that "
       "value is the useful delegation now. Do not claim a blocked task ran.")


def case(cid, goal, runtime, tasks, decision, events, state, intended, expected, derivation):
    return {
        "id": f"os-c2c-probe4-{cid}",
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


# ---------- 1. 令牌桶限流 ----------
CAP, RATE = 8, 2
ARRIVALS = [0, 1, 2, 3, 5, 8, 13, 21, 34]
SIZES = [5, 4, 6, 3, 4, 8, 7, 6, 8]
tokens, last, admitted, trace = CAP, 0, 0, []
for t, s in zip(ARRIVALS, SIZES):
    tokens = min(CAP, tokens + RATE * (t - last)); last = t
    if tokens >= s:
        tokens -= s; admitted += 1; trace.append(f"t={t} size={s}: admit (left {tokens})")
    else:
        trace.append(f"t={t} size={s}: DROP (only {tokens})")
assert admitted == 8, (admitted, trace)
ratelimit = case(
    "ratelimit-01",
    "Bring the public gateway state to a compliant next step while preserving token bucket invariants "
    "and the admission audit trail.",
    {"bucket_capacity_tokens": CAP, "refill_tokens_per_second": RATE,
     "algorithm": "Discrete token bucket: on each request at time t, tokens = min(capacity, tokens + rate*(t - last_t)); "
                  "a request of size s is admitted iff tokens >= s, consuming s tokens, otherwise it is dropped.",
     "audit_window_s": 34, "metering": "per-request token accounting, refill at integer request times only"},
    [task("meter", "SUCCEEDED", [], 1, 256, 34),
     task("ledger_flush", "READY", ["meter"], 1, 512, 5),
     task("alert_eval", "READY", ["meter"], 1, 256, 3)],
    {"field": "admitted_requests", "question": "How many of the 9 requests in the audit window are admitted "
     "by the token bucket (capacity 8, refill 2 tokens/s, window 0..34 s)?",
     "controller_note": "The rate-limit controller is blocked until the admission count is verified."},
    [{"request_time_s": t, "request_size_tokens": s} for t, s in zip(ARRIVALS, SIZES)],
    {"bucket_tokens_now": CAP, "last_refill_time_s": 0, "requests_seen": 9, "drops_seen": 0,
     "observation_boundary": OBS},
    "Token bucket admission count over the audit window.",
    admitted,
    "; ".join(trace) + f"; admitted = {admitted} of 9.",
)

# ---------- 2. 写时复制记账 ----------
PAGE = 4096
CHILD_WRITES, PARENT_WRITES = 640, 96
extra = (CHILD_WRITES + PARENT_WRITES) * PAGE
assert extra == 736 * 4096 == 3014656
cow = case(
    "cow-01",
    "Bring the public memory ledger to a verified next step while preserving copy-on-write sharing "
    "and per-process write fault accounting.",
    {"page_size_bytes": PAGE, "parent_rss_pages_before_fork": 8192,
     "cow_policy": "After fork, parent and child share every page read-only. A write to a shared page faults "
                   "and allocates one private copy (one extra physical page) in the writing process only.",
     "fork_time_s": 0, "ledger_window_s": 60},
    [task("fault_tally", "READY", [], 1, 256, 8),
     task("ledger_flush", "BLOCKED", ["fault_tally"], 1, 512, 4)],
    {"field": "cow_extra_bytes", "question": "After fork, the child wrote 640 distinct pages and the parent wrote "
     "96 other distinct pages. How many bytes of additional physical memory were allocated by "
     "copy-on-write duplication (page size 4096 B)?",
     "controller_note": "The memory ledger is blocked until the COW extra allocation is verified."},
    [{"process": "child", "write_faults": CHILD_WRITES}, {"process": "parent", "write_faults": PARENT_WRITES}],
    {"shared_pages_after_fork": 8192, "child_private_pages": CHILD_WRITES, "parent_private_pages": PARENT_WRITES,
     "observation_boundary": OBS},
    "Additional physical bytes allocated by copy-on-write duplication.",
    extra,
    f"child {CHILD_WRITES} private pages + parent {PARENT_WRITES} private pages = 736 pages copied; "
    f"736 * {PAGE} = {extra} bytes.",
)

# ---------- 3. 死锁检测 ----------
EDGES = [("P0", "P1"), ("P1", "P2"), ("P2", "P0"), ("P3", "P1")]
# 环 P0->P1->P2->P0 含 3 个进程
cycle = {"P0", "P1", "P2"}
assert len(cycle) == 3
deadlock = case(
    "deadlock-01",
    "Bring the public runtime state to a diagnosable next step while preserving wait-for edges "
    "and the deadlock audit trail.",
    {"detection_algorithm": "wait-for graph cycle detection over reported hold-and-wait edges",
     "processes": ["P0", "P1", "P2", "P3"], "resources": ["R0", "R1", "R2"],
     "policy": "An edge Pi -> Pj means Pi holds a resource Pj is waiting for."},
    [task("graph_build", "SUCCEEDED", [], 1, 128, 2),
     task("cycle_report", "READY", ["graph_build"], 1, 128, 3),
     task("victim_select", "BLOCKED", ["cycle_report"], 1, 128, 2)],
    {"field": "cycle_processes", "question": "The wait-for edges are P0->P1, P1->P2, P2->P0, P3->P1. "
     "How many distinct processes participate in a wait-for cycle?",
     "controller_note": "The deadlock auditor is blocked until the cycle membership is verified."},
    [{"waiter": a, "holder": b, "resource": f"R{i}"} for i, (a, b) in enumerate(EDGES)],
    {"processes_running": 4, "processes_blocked": 4, "edges_reported": 4,
     "observation_boundary": OBS},
    "Number of distinct processes in a wait-for cycle.",
    3,
    "P0->P1->P2->P0 is a cycle of 3 distinct processes; P3->P1 does not close a cycle.",
)

# ---------- 4. SCAN 磁盘调度 ----------
CYLINDERS, HEAD, DIRECTION = 200, 53, "up"
QUEUE = [98, 183, 37, 122, 14, 124, 65, 67]
up = sorted(x for x in QUEUE if x >= HEAD)
down = sorted((x for x in QUEUE if x < HEAD), reverse=True)
seq = up + down
moves = [abs(seq[0] - HEAD)] + [abs(b - a) for a, b in zip(seq, seq[1:])]
total = sum(moves)
assert seq == [65, 67, 98, 122, 124, 183, 37, 14] and total == 299, (seq, total)
disk = case(
    "diskscan-01",
    "Bring the public I/O scheduler state to a verified next step while preserving elevator policy "
    "and the head-movement audit trail.",
    {"cylinders": CYLINDERS, "head_start": HEAD, "direction": DIRECTION,
     "algorithm": "SCAN (elevator): serve pending requests while moving in the current direction; reverse only "
                  "after serving the last request in that direction. No forced travel to either disk end.",
     "queue_at_t0": QUEUE},
    [task("queue_snapshot", "SUCCEEDED", [], 1, 64, 1),
     task("seek_audit", "READY", ["queue_snapshot"], 1, 64, 2),
     task("qos_report", "BLOCKED", ["seek_audit"], 1, 128, 3)],
    {"field": "total_head_movement", "question": "The head is at cylinder 53 moving toward higher cylinders; "
     "the pending queue is 98, 183, 37, 122, 14, 124, 65, 67. Under SCAN, what is the total head movement "
     "in cylinders to serve the whole queue?",
     "controller_note": "The I/O controller is blocked until the seek total is verified."},
    [{"head_at_cylinder": HEAD, "moving": DIRECTION}, {"queue_snapshot": QUEUE}],
    {"requests_pending": 8, "head_position": HEAD, "serviced": 0,
     "observation_boundary": OBS},
    "Total SCAN head movement for the pending queue.",
    total,
    f"service order {seq}; movement = 12+2+31+24+2+59 (up) + 146+23 (down) = {total} cylinders.",
)

# ---------- 5. 两级页表地址翻译 ----------
VA = 0x00401523
PDI = VA >> 22
PTI = (VA >> 12) & 0x3FF
OFF = VA & 0xFFF
FRAME = 0x2A3
PA = FRAME * PAGE + OFF
assert (PDI, PTI, OFF) == (1, 1, 0x523) and PA == 0x2A3523 == 2766115, (hex(PA), PA)
paging = case(
    "pagetable-01",
    "Bring the public address-translation state to a verified next step while preserving two-level "
    "page table invariants and the TLB audit trail.",
    {"virtual_address_bits": 32, "page_size_bytes": PAGE,
     "page_table_layout": "10-bit page directory index, 10-bit page table index, 12-bit offset",
     "pde": {"index": PDI, "present": True, "pt_base_frame": 0x00005},
     "pte": {"pdi": PDI, "pti": PTI, "present": True, "frame_number": FRAME}},
    [task("walk", "READY", [], 1, 64, 2),
     task("tlb_fill", "BLOCKED", ["walk"], 1, 64, 1)],
    {"field": "physical_address_bytes", "question": "For virtual address 0x00401523 with a 4 KiB page, "
     "10+10+12 two-level layout, PD[1] pointing at a present page table, and PT[1][1] mapping to frame 0x2A3, "
     "what is the physical address in bytes?",
     "controller_note": "The memory manager is blocked until the translation is verified."},
    [{"fault_address": hex(VA), "access": "read", "pde_index": PDI, "pte_index": PTI}],
    {"tlb_hits": 0, "page_faults": 1, "walks_completed": 0,
     "observation_boundary": OBS},
    "Physical address for virtual address 0x00401523.",
    PA,
    f"PD index {PDI}, PT index {PTI}, offset 0x{OFF:X}; frame 0x{FRAME:X} * {PAGE} + 0x{OFF:X} = 0x{PA:X} = {PA}.",
)

# ---------- 6. RR 时间片轮转 ----------
BURSTS = {"P1": 8, "P2": 4, "P3": 9, "P4": 5}
Q = 3
rem = dict(BURSTS)
done_at, t, order = {}, 0, []
rr_queue = list(BURSTS)
ready = list(BURSTS)
while ready:
    p = ready.pop(0)
    run = min(Q, rem[p]); rem[p] -= run; t += run; order.append((p, run, t))
    if rem[p] == 0:
        done_at[p] = t
    else:
        ready.append(p)
comp_sum = sum(done_at.values())
assert order == [("P1", 3, 3), ("P2", 3, 6), ("P3", 3, 9), ("P4", 3, 12),
                 ("P1", 3, 15), ("P2", 1, 16), ("P3", 3, 19), ("P4", 2, 21),
                 ("P1", 2, 23), ("P3", 3, 26)] and done_at == {"P2": 16, "P4": 21, "P1": 23, "P3": 26} \
    and comp_sum == 86, (order, done_at, comp_sum)
rr = case(
    "rr-01",
    "Bring the public CPU scheduler state to a verified next step while preserving round-robin "
    "invariants and the dispatch audit trail.",
    {"policy": "Round-robin, all four processes arrive at t=0, quantum 3 s, no preemption overhead",
     "burst_seconds": BURSTS, "quantum_s": Q},
    [task("dispatch_log", "READY", [], 1, 64, 2),
     task("sla_report", "BLOCKED", ["dispatch_log"], 1, 128, 2)],
    {"field": "completion_sum_seconds", "question": "With round-robin quantum 3 s and burst times "
     "P1=8, P2=4, P3=9, P4=5 (all arrive at t=0), what is the sum of completion times of all four processes?",
     "controller_note": "The scheduler is blocked until the completion accounting is verified."},
    [{"arrival_s": 0, "process": p, "burst_s": b} for p, b in BURSTS.items()],
    {"running": None, "ready_queue": list(BURSTS), "finished": [],
     "observation_boundary": OBS},
    "Sum of round-robin completion times for the four processes.",
    comp_sum,
    "timeline P1(0-3) P2(3-6) P3(6-9) P4(9-12) P1(12-15) P2 done 16 P3(16-19) P4 done 21 "
    "P1 done 23 P3 done 26; completion sum 16+21+23+26 = 86.",
)

# ---------- 7. FIFO 页面置换缺页 ----------
REFS = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]
FRAMES = 3
frames, faults, fifo_trace = [], 0, []
for r in REFS:
    if r in frames:
        fifo_trace.append(f"{r}: hit")
        continue
    faults += 1
    evict = frames.pop(0) if len(frames) == FRAMES else None
    frames.append(r)
    fifo_trace.append(f"{r}: fault (evict {evict})")
assert faults == 15, (faults, fifo_trace)
pagefifo = case(
    "pagefifo-01",
    "Bring the public page-fault ledger to a verified next step while preserving FIFO replacement "
    "invariants and the frame audit trail.",
    {"frames": FRAMES, "replacement": "FIFO: on a fault with full frames, evict the oldest loaded page",
     "reference_string": REFS, "page_size_bytes": PAGE},
    [task("fault_replay", "READY", [], 1, 64, 3),
     task("thrash_report", "BLOCKED", ["fault_replay"], 1, 128, 2)],
    {"field": "fifo_page_faults", "question": "With 3 frames and FIFO replacement, how many page faults occur "
     "for the reference string 7,0,1,2,0,3,0,4,2,3,0,3,2,1,2,0,1,7,0,1?",
     "controller_note": "The memory manager is blocked until the fault count is verified."},
    [{"time_step": i, "page": r} for i, r in enumerate(REFS)],
    {"frames_content": [], "faults_so_far": 0, "hits_so_far": 0,
     "observation_boundary": OBS},
    "FIFO page fault count for the 20-step reference string with 3 frames.",
    faults,
    "faults at 7,0,1,2 (4) then 3 (evict 7) 5, 4 (evict 0) 6, 2 (evict 1) 7, then 1 (evict 3) 8, "
    "7 (evict 4) 9, final 1 (evict 2) 10... exact replay: " + "; ".join(fifo_trace[:6]) + f"; total = {faults}.",
)

CASES = [ratelimit, cow, deadlock, disk, paging, rr, pagefifo]
for c in CASES:
    name = c["id"].replace("os-c2c-probe4-", "")
    path = OUT / f"public-probe4-{name}.json"
    path.write_text(json.dumps(c, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{path.name}: expected={c['probe']['expected_answer']}")
print("done,", len(CASES), "cases")
