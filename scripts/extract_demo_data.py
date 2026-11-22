"""从 probe4/probe5/acbi4 批次提取演示网页数据 → demo/delegation-walkthrough/src/demoData.ts"""
import json, re
from pathlib import Path

ART = Path(r"E:/code/c2cos/artifacts/probe4-batch-20261020")
OUT = Path(r"E:/code/c2cos/demo/delegation-walkthrough/src/demoData.ts")

ARM_LABELS = {
    "c2c_hidden_analysis": "c2c_hidden_analysis（KV 隐藏融合）",
    "c2c_shared": "c2c_shared（KV 融合）",
    "text_full": "text_full（全文文字）",
    "text_brief": "text_brief（简报文字）",
    "c2c_hidden_full": "c2c_hidden_full（全转录隐藏）",
    "c2c_state_hidden": "c2c_state_hidden（状态隐藏）",
    "state_native": "state_native（原生状态）",
    "c2c_command": "c2c_command（仅命令）",
}


def load(p):
    return json.loads((ART / p).read_text(encoding="utf-8"))


def probe_scenario(run_dir, case_file, title, subtitle, scene_steps, think_note):
    r = load(f"{run_dir}/result.json")
    case = json.loads(Path(f"E:/code/c2cos/data/c2c-core-os/{case_file}").read_text(encoding="utf-8"))
    expected = case["probe"]["expected_answer"]
    pa = (r.get("planner_action") or {})
    tasks = pa.get("tasks") or []
    cmd = tasks[0].get("command", "") if tasks else ""
    reason = pa.get("reason", "")
    note = ""
    p = r.get("planner") or {}
    m = (p.get("metrics") or {})
    for s in (r.get("planner_stages") or []):
        o = s.get("output") or ""
        if "planner_working_note" in o:
            try:
                note = json.loads(o).get("planner_working_note", "")
            except Exception:
                note = o[:400]
    arms = []
    for name, a in (r.get("arms") or {}).items():
        out = a.get("output") or ""
        correct = False
        try:
            j = json.loads(out)
            rep = json.dumps(j, ensure_ascii=False)
            correct = bool(re.search(rf"(?<![\d.]){expected}(?![\d.])", rep))
        except Exception:
            correct = bool(re.search(rf"(?<![\d.]){expected}(?![\d.])", out))
        arms.append({
            "name": name,
            "label": ARM_LABELS.get(name, name),
            "correct": correct,
            "decodeS": round(a.get("decode_seconds") or 0, 2),
            "hiddenTokens": a.get("hidden_context_tokens") or 0,
            "consumedFused": bool(a.get("worker_consumed_fused_cache")),
        })
    return {
        "id": run_dir.replace("probe4-", "").replace("probe5-", ""),
        "kind": "probe",
        "title": title,
        "subtitle": subtitle,
        "expected": expected,
        "derivation": case["probe"]["derivation"],
        "question": case["public_background"]["pending_decision"]["question"],
        "command": cmd,
        "reason": reason,
        "note": note,
        "wallS": round(r.get("wall_seconds") or 0, 1),
        "arms": arms,
        "sceneSteps": scene_steps,
        "thinkNote": think_note,
    }


# ---------- acbi4 全设计批 ----------
r = load("acbi4-osclassics-fault-01/result.json")
a = r.get("acb_report") or {}
tasks = a.get("tasks") or {}

def acb_task(tid, field):
    t = load(f"acbi4-osclassics-fault-01/tasks/{tid}.json")
    gate = t.get("application_gate") or {}
    out = (t.get("output") or "")
    val = None
    try:
        j = json.loads(out[out.index("{"):])
        rep = j.get("report") or {}
        if field in rep:
            val = rep[field]
        res = rep.get("result")
        if val is None and isinstance(res, dict) and field in res:
            val = res[field]
    except Exception:
        m = re.search(r'"%s"\s*:\s*(\d+)' % re.escape(field), out)
        val = m.group(1) if m else None
    return {
        "decodeS": round(t.get("decode_seconds") or 0, 1),
        "prefillS": round(t.get("receiver_task_prefill_seconds") or 0, 3),
        "hiddenTokens": t.get("hidden_context_tokens"),
        "gateReason": gate.get("reason", ""),
        "value": str(val) if val is not None else "",
        "outputTokens": t.get("output_tokens"),
        "consumedFused": bool(t.get("worker_consumed_fused_cache")),
        "peakIncMib": round(t.get("peak_incremental_mib") or 0, 1),
    }

planner_json = {}
p = r.get("planner") or {}
try:
    planner_json = json.loads(p.get("output") or "{}")
except Exception:
    pass
note = planner_json.get("planner_working_note", "")
reason = planner_json.get("reason", "")

fault = {
    "id": "fault-isolated",
    "kind": "acb",
    "title": "内存限额记账 · 全设计贯通版",
    "subtitle": "osclassics-fault-01 · acbi4 批次 · 隔离子进程 + cgroup 限额 + mmap 零拷贝 + LRU 池 + C2C",
    "arm": "c2c_hidden_analysis",
    "seed": 42,
    "wallS": round(r.get("wall_seconds") or 0, 1),
    "memoryLimitMb": 4096,
    "quota": {"tokens": 400, "memoryMib": 512},
    "poolCapacity": 1,
    "note": note,
    "reason": reason,
    "agents": [
        {"id": "A1", "exitCode": 0, "peakRssMb": 1024},
        {"id": "A2", "exitCode": 137, "peakRssMb": 8200},
        {"id": "A3", "exitCode": 0, "peakRssMb": 2340},
        {"id": "A4", "exitCode": 0, "peakRssMb": 1788},
    ],
    "tasks": {
        "t1": {"label": "t1 · 计算被杀代理序号",
               "command": "compute oom_agent_index as the index of the agent with exit code 137 and peak_rss_mb > 4096",
               "deliverable": "oom_agent_index", "success": "oom_agent_index == 1",
               **acb_task("t1", "oom_agent_index")},
        "t2": {"label": "t2 · 计算存活兄弟数",
               "command": "compute survived_siblings as the count of agents with exit code 0",
               "deliverable": "survived_siblings", "success": "survived_siblings == 3",
               **acb_task("t2", "survived_siblings")},
    },
    "poolEvents": [
        "%s(%s)" % (e.get("event", "?"), e.get("task_id") or e.get("task", "?"))
        for e in (r.get("payload_pool_events") or [])
    ],
    "events": [
        {"event": e.get("event"), "task": e.get("task") or e.get("task_id") or "",
         "detail": (e.get("error") or (e.get("gate") or {}).get("reason") or
                    e.get("detail") or (e.get("gate") or {}).get("accepted") and "accepted" or "")}
        for e in (r.get("scheduler_events") or [])
    ],
    "plannerWallS": round((p.get("metrics") or {}).get("generation_seconds") or 0, 1),
    "plannerTokens": (p.get("metrics") or {}).get("generated_tokens"),
    "kvMb": 260,
}

deadlock = probe_scenario(
    "probe4-deadlock-01", "public-probe4-deadlock-01.json",
    "死锁检测 · 等待环记账",
    "os-c2c-probe4-deadlock-01 · probe4 批次 · 8/8 臂全部命中（本批最佳实例）",
    [
        "等待边已上报：P0→P1、P1→P2、P2→P0、P3→P1（Pi→Pj 表示 Pi 持有 Pj 等待的资源）。",
        "检测器问题：有几个不同的进程处于等待环中？",
    ],
    "P0→P1→P2→P0 构成环，含 3 个不同进程；P3→P1 不构成环。答案 3。",
)

pagetable = probe_scenario(
    "probe5-pagetable-01", "public-probe5-pagetable-01.json",
    "页表地址翻译 · 缺页现场",
    "os-c2c-probe5-pagetable-01 · probe5 批次 · C2C 隐藏臂命中，与文字臂同档",
    [
        "一次读访问触发缺页：虚拟地址 34083，页大小 4096 B，映射到物理帧 8（present）。",
        "内存管理器问题：物理地址是多少字节？（翻译规则：帧号 × 页大小 + 页内偏移）",
    ],
    "帧 8 × 4096 = 32768，加偏移 1315（34083 − 32768）= 34083。答案 34083。",
)

ratelimit = probe_scenario(
    "probe5-ratelimit-01", "public-probe5-ratelimit-01.json",
    "令牌桶限流 · 准入记账",
    "os-c2c-probe5-ratelimit-01 · probe5 批次 · C2C 隐藏臂命中，与文字臂同档",
    [
        "令牌桶：容量 10，补充速率 1 token/s；窗口内 4 个请求：t=0 大小 3，t=1 大小 2，t=2 大小 4，t=3 大小 5。",
        "限流控制器问题：有几个请求被准入？",
    ],
    "t=0 准入余 7；t=1 准入余 6；t=2 准入余 3；t=3 桶内 4 < 5 丢弃。准入 3 个。答案 3。",
)

ts = "export interface DemoData {\n  fault: any; deadlock: any; pagetable: any; ratelimit: any;\n}\n"
ts += "export const DEMO: DemoData = " + json.dumps(
    {"fault": fault, "deadlock": deadlock, "pagetable": pagetable, "ratelimit": ratelimit},
    ensure_ascii=False, indent=2) + ";\n"
OUT.write_text(ts, encoding="utf-8")
print("written", OUT, len(ts), "bytes")
print("deadlock arms:", [(x['name'], x['correct']) for x in deadlock['arms']])
print("pagetable arms:", [(x['name'], x['correct']) for x in pagetable['arms']])
print("ratelimit arms:", [(x['name'], x['correct']) for x in ratelimit['arms']])
print("fault t1:", fault['tasks']['t1'])
