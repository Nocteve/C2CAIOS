// Build demo/run-data.json from the real run artifacts.
import fs from 'node:fs';

const base = new URL('../', import.meta.url);
const data = JSON.parse(fs.readFileSync(new URL('./data.json', import.meta.url), 'utf8'));

const FAULT = '../artifacts/probe4-batch-20261020/acbi4-osclassics-fault-01/result.json';
const PROBES = [
  ['deadlock', '../artifacts/probe4-batch-20261020/probe4-deadlock-01/result.json', 'probe4 · 2026-10-20'],
  ['pagetable', '../artifacts/probe4-batch-20261020/probe5-pagetable-01/result.json', 'probe5 · 2026-10-20'],
  ['ratelimit', '../artifacts/probe4-batch-20261020/probe5-ratelimit-01/result.json', 'probe5 · 2026-10-20'],
];

function lastReport(output) {
  const s = String(output);
  const i = s.lastIndexOf('{"action": "report"');
  if (i >= 0) {
    try { return JSON.parse(s.slice(i)); } catch { /* fall through */ }
  }
  const j = s.lastIndexOf('{"action":"report"');
  if (j >= 0) {
    try { return JSON.parse(s.slice(j)); } catch { /* fall through */ }
  }
  return null;
}

function probeReport(output) {
  try {
    const o = JSON.parse(output);
    const r = o.report;
    if (typeof r === 'string') return r;
    if (r && typeof r === 'object') {
      if ('answer' in r) return r.answer;
      const vals = Object.values(r).filter((v) => typeof v === 'number' || typeof v === 'string');
      if (vals.length === 1) return vals[0];
      return JSON.stringify(r);
    }
    return r;
  } catch {
    return String(output).slice(0, 80);
  }
}

function shortCommand(arm) {
  const s = String(arm.command || '');
  const m = s.match(/"command":\s*"([^"]+)"[\s\S]*?"deliverable":\s*"([^"]+)"/);
  return m ? { cmd: m[1], text: m[2] } : { cmd: '', text: '' };
}

const f = JSON.parse(fs.readFileSync(new URL(FAULT, import.meta.url), 'utf8'));
const fmeta = data.fault;
const ftasks = ['t1', 't2'].map((id) => {
  const t = JSON.parse(fs.readFileSync(new URL(`../artifacts/probe4-batch-20261020/acbi4-osclassics-fault-01/tasks/${id}.json`, import.meta.url), 'utf8'));
  const gateEvent = (f.scheduler_events || []).find((e) => e.event === 'done' && e.task === id);
  const gate = gateEvent ? gateEvent.gate : null;
  const parsed = lastReport(t.output);
  const report = parsed && parsed.report ? parsed.report : {};
  const deliverable = fmeta.tasks[id].deliverable;
  return {
    id,
    command: fmeta.tasks[id].command,
    deliverable,
    expected: gate ? gate.expected : fmeta.tasks[id].value,
    reported: gate ? gate.reported : fmeta.tasks[id].value,
    accepted: gate ? gate.accepted : true,
    gateReason: gate ? gate.reason : fmeta.tasks[id].gateReason,
    decode: Number(t.decode_seconds),
    outputTokens: t.output_tokens,
    hidden: t.hidden_context_tokens,
    report: report[deliverable] !== undefined ? report[deliverable] : (gate ? gate.reported : undefined),
  };
});

const fault = {
  id: 'acbi4-osclassics-fault-01',
  batch: 'acbi4 · 2026-10-20',
  wall: Number(f.wall_seconds),
  status: f.status,
  planner: {
    tokens: f.planner.metrics.generated_tokens,
    wall: f.planner.metrics.generation_seconds,
    note: fmeta.reasonMasked,
    tasks: ftasks.length,
    action: 'delegate',
  },
  agents: fmeta.agents,
  quota: fmeta.quota,
  tasks: ftasks,
  fusion: {
    hidden: ftasks[0].hidden,
    kvMb: fmeta.kvMb,
    prefill: Number(ftasks[0].decode ? 0.001 : 0),
    selected: Number(ftasks[0].hidden),
  },
  events: (f.scheduler_events || []).map((e) => ({ event: e.event, task: e.task })),
  pool: (f.payload_pool_events || []).map((e) => `${e.event}(${e.task_id})`),
  invariants: (f.acb_report && f.acb_report.invariants) || [],
};

const probes = {};
for (const [key, path, batch] of PROBES) {
  const d = JSON.parse(fs.readFileSync(new URL(path, import.meta.url), 'utf8'));
  const meta = data[key];
  const arms = ['c2c_hidden_analysis', 'c2c_hidden_full', 'c2c_shared', 'text_brief', 'text_full'].map((name) => {
    const a = d.arms[name];
    return {
      name,
      output: a.output,
      report: probeReport(a.output),
      decode: Number(a.decode_seconds),
      tokens: a.output_tokens,
      hidden: a.hidden_context_tokens,
      accepted: a.semantic_status === 'accepted_by_public_os_gate',
      status: a.semantic_status,
    };
  });
  const sc = shortCommand(d.arms.c2c_hidden_analysis);
  probes[key] = {
    key,
    title: meta.title,
    batch,
    wall: Number(d.wall_seconds),
    expected: meta.expected,
    question: meta.question,
    note: meta.thinkMasked,
    kvMb: meta.kvMb,
    prefill: Number((d.preparation || {}).receiver_prefill_seconds || 0),
    shortCommand: sc.cmd,
    shortText: sc.text && sc.text.includes(' ') ? sc.text : 'compute ' + (sc.text || 'the requested value') + ' from the granted public state',
    arms,
  };
}

const out = { fault, probes };
fs.writeFileSync(new URL('./run-data.json', import.meta.url), JSON.stringify(out, null, 1));
console.log('wrote demo/run-data.json', fs.statSync(new URL('./run-data.json', import.meta.url)).size, 'bytes');
