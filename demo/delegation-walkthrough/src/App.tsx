import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Cpu,
  MemoryStick,
  FileJson,
  ArrowRight,
  CheckCircle2,
  ShieldCheck,
  GitBranch,
  HardDrive,
  Timer,
  Lock,
  Database,
  Terminal,
  Zap,
} from 'lucide-react';
import { DEMO } from './demoData';

/* ---------- 打字机 ---------- */
function useTypewriter(text: string, active: boolean, cps = 90) {
  const [n, setN] = useState(0);
  useEffect(() => {
    if (!active) return;
    setN(0);
    const t = setInterval(() => {
      setN((v) => {
        if (v >= text.length) {
          clearInterval(t);
          return v;
        }
        return v + 1;
      });
    }, 1000 / cps);
    return () => clearInterval(t);
  }, [text, active, cps]);
  return active ? text.slice(0, n) : text;
}

/* ---------- 展示层答案脱敏：判分数值在揭晓前以 ··· 代替 ---------- */
function maskAnswer(s: string): string {
  return (s || '').replace(/(={1,2}\s*)\d+/g, '$1··· ');
}

/* ---------- 运行时事件流（fault 场景真实事件记录） ---------- */
function EventLog({ stepIdx, total }: { stepIdx: number; total: number }) {
  const f = DEMO.fault;
  const progress = total > 1 ? stepIdx / (total - 1) : 1;
  const visible = Math.max(0, Math.ceil(progress * f.events.length));
  const color = (e: string) =>
    e === 'submit' ? 'text-slate-400' : e === 'admit' ? 'text-sky-400' : 'text-emerald-400';
  return (
    <div className="rounded-xl border bg-card overflow-hidden">
      <div className="px-3 py-2 border-b text-[11px] text-muted-foreground flex items-center gap-1.5 font-mono">
        <Terminal className="w-3.5 h-3.5" /> runtime event log · 运行时事件流（真实运行记录）
      </div>
      <div className="bg-slate-950 px-3 py-2.5 font-mono text-[11px] leading-6 min-h-[132px]">
        {f.events.slice(0, visible).map((e: any, i: number) => (
          <div key={i} className="flex gap-2 whitespace-nowrap overflow-hidden">
            <span className="text-slate-500 tabular-nums shrink-0">{(i * 0.7).toFixed(1).padStart(5, '0')}s</span>
            <span className={`shrink-0 ${color(e.event)}`}>[{e.event.toUpperCase()}]</span>
            <span className="text-slate-200 shrink-0">{e.task}</span>
            {e.detail ? <span className="text-slate-500 truncate">· {e.detail}</span> : null}
          </div>
        ))}
        <span className={`text-emerald-400 ${visible >= f.events.length ? 'opacity-0' : 'animate-caret'}`}>▌</span>
      </div>
    </div>
  );
}

/* ---------- 左栏：ACB 场景现场 ---------- */
function FaultPanel({ stepIdx, steps }: { stepIdx: number; steps: StepDef[] }) {
  const f = DEMO.fault;
  const at = (id: string) => stepIdx >= steps.findIndex((s) => s.id === id);
  return (
    <div className="rounded-xl border bg-card p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="text-sm font-medium flex items-center gap-2">
          <MemoryStick className="w-4 h-4 text-amber-500" />
          运行现场 · memory.max
        </div>
        <span className="text-xs font-mono text-amber-600 dark:text-amber-400 border border-amber-500/40 rounded-full px-2 py-0.5">
          {f.memoryLimitMb} MB
        </span>
      </div>
      <div className="space-y-1.5">
        {f.agents.map((a: any) => {
          const killed = a.exitCode === 137;
          const revealed = at('scene');
          const gateOpen = at('gate');
          return (
            <div
              key={a.id}
              className={`grid grid-cols-[36px_64px_1fr] items-center gap-2 rounded-lg border px-2.5 py-1.5 text-xs transition-all duration-700 ${
                killed && gateOpen ? 'border-red-500/50 bg-red-500/10' : 'border-border bg-background/60'
              }`}
            >
              <span className="font-mono font-medium">{a.id}</span>
              {revealed ? (
                <>
                  <span className={`font-mono ${killed && gateOpen ? 'text-red-600 dark:text-red-400' : 'text-emerald-600 dark:text-emerald-400'}`}>
                    rc={a.exitCode}
                  </span>
                  <span className="text-muted-foreground font-mono">
                    peak {a.peakRssMb} MB{killed && gateOpen ? ' · 超限被杀' : ''}
                  </span>
                </>
              ) : (
                <span className="col-span-2 text-muted-foreground/60 font-mono">running…</span>
              )}
            </div>
          );
        })}
      </div>
      {/* OS 机制徽章 */}
      <div className="border-t pt-3 space-y-2">
        <div className="text-[11px] font-medium text-muted-foreground flex items-center gap-1.5">
          <ShieldCheck className="w-3.5 h-3.5" /> 本实例激活的 OS 机制（全部真实执行）
        </div>
        {[
          ['隔离子进程', '每任务以真实 OS 子进程执行，消费 mmap 发布的融合 KV', at('c2c')],
          ['worker 限额（模拟 cgroup）', `memory ≤ ${f.quota.memoryMib} MiB · pids ≤ ${f.quota.pidsMax} · ${f.quota.timeoutS}s（resource_control 记账：t1 峰值 191.5 MiB/323 tok，零违规）`, at('c2c')],
          ['mmap 零拷贝', '载荷以 np.memmap 装载，host 侧零拷贝', at('c2c')],
          [`LRU 池（容量 ${f.poolCapacity}）`, f.poolEvents.length ? `置换事件：${f.poolEvents.join('、')}` : '容量 1，跨任务触发 swap-out', at('t2')],
        ].map(([t, d, on]: any) => (
          <div key={t} className={`flex items-start gap-2 text-[11px] transition-opacity duration-700 ${on ? 'opacity-100' : 'opacity-35'}`}>
            <Lock className="w-3.5 h-3.5 mt-0.5 text-emerald-600 dark:text-emerald-400 shrink-0" />
            <div>
              <div className="font-medium">{t}</div>
              <div className="text-muted-foreground">{d}</div>
            </div>
          </div>
        ))}
      </div>
      <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground pt-1 border-t">
        <FileJson className="w-3.5 h-3.5" />
        runtime-state.json · 公开状态（Worker 可见）
      </div>
    </div>
  );
}

/* ---------- 左栏：probe 场景现场 ---------- */
function ProbePanel({ scenario, stepIdx }: { scenario: any; stepIdx: number }) {
  const revealed = stepIdx >= 0;
  const answerRevealed = stepIdx >= 4;
  return (
    <div className="rounded-xl border bg-card p-4 space-y-3">
      <div className="text-sm font-medium flex items-center gap-2">
        <HardDrive className="w-4 h-4 text-amber-500" />
        场景现场
      </div>
      <div className="space-y-1.5">
        {scenario.sceneSteps.map((s: string, i: number) => (
          <div
            key={i}
            className={`rounded-lg border px-2.5 py-2 text-xs leading-relaxed transition-all duration-700 ${
              revealed ? 'border-border bg-background/60 opacity-100' : 'opacity-40'
            }`}
          >
            {s}
          </div>
        ))}
      </div>
      <div className="border-t pt-3 space-y-1.5 text-[11px]">
        <div className="text-muted-foreground font-medium flex items-center gap-1.5">
          <Database className="w-3.5 h-3.5" /> 判分口径（离线{answerRevealed ? '' : ' · 答案在第 5 步揭晓'}）
        </div>
        <div className="font-mono text-muted-foreground break-all">
          {answerRevealed ? `expected_answer = ${scenario.expected}` : 'expected_answer = ···（判分步骤揭晓）'}
        </div>
        <div className="text-muted-foreground leading-relaxed">{scenario.derivationMasked || scenario.derivation}</div>
      </div>
    </div>
  );
}

/* ---------- 舞台步骤：ACB ---------- */
function StepFaultScene() {
  const f = DEMO.fault;
  return (
    <div className="space-y-3">
      <p className="text-sm leading-relaxed text-muted-foreground">
        四个代理 A1–A4 在 <b className="text-foreground">{f.memoryLimitMb} MB</b> 内存限额下运行，结束后要回答：
        谁被限额处决（<span className="font-mono text-foreground">oom_agent_index</span>）？几个兄弟存活（
        <span className="font-mono text-foreground">survived_siblings</span>）？
        本实例与基础版不同：每个 Worker 任务将以<b className="text-foreground">真实 OS 子进程</b>在 cgroup 限额与 LRU 载荷池下执行。
      </p>
      <div className="rounded-lg border border-dashed p-3 text-xs text-muted-foreground font-mono">
        $ cat runtime-state.json → 左侧面板逐代理揭示 exit code 与 peak RSS
      </div>
    </div>
  );
}

function StepPlan({ active, note, stats }: { active: boolean; note: string; stats: [string, string][] }) {
  const typed = useTypewriter(note, active, 80);
  return (
    <div className="space-y-3">
      <div className="rounded-xl border-2 border-dashed p-4 bg-muted/40">
        <span className="inline-flex items-center gap-1.5 text-[11px] text-amber-600 dark:text-amber-400 border border-amber-500/40 rounded-full px-2 py-0.5 mb-3">
          <ShieldCheck className="w-3 h-3" />
          Planner 工作笔记 · 永不以文字发送，只以 KV 形式存在
        </span>
        <p className="font-mono text-[13px] leading-relaxed min-h-[100px]">{typed}</p>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        {stats.map(([v, l]) => (
          <div key={l} className="rounded-lg border bg-card px-3 py-2">
            <div className="text-lg font-semibold tabular-nums">{v}</div>
            <div className="text-[11px] text-muted-foreground">{l}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function StepC2c({ active, hiddenTokens, kvMb, extra }: { active: boolean; hiddenTokens: number; kvMb: number; extra?: string }) {
  return (
    <div className="space-y-4">
      <div className="flex items-stretch gap-2">
        <div className="flex-1 rounded-xl border bg-card p-3">
          <div className="text-xs font-medium flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5" /> Planner KV 池
          </div>
          <div className="text-[11px] text-muted-foreground mt-1 leading-relaxed">
            笔记完成编码 · 选区 {hiddenTokens} tokens 被发布
          </div>
        </div>
        <div className="self-center text-center w-28 shrink-0">
          <ArrowRight className={`w-5 h-5 mx-auto ${active ? 'text-emerald-500 animate-kv-flow' : 'text-muted-foreground'}`} />
          <div className="text-[10px] text-muted-foreground mt-0.5">fuser 选区融合</div>
          <div className="text-xs font-mono font-medium text-emerald-600 dark:text-emerald-400">{hiddenTokens} tokens</div>
        </div>
        <div className="flex-1 rounded-xl border bg-card p-3">
          <div className="text-xs font-medium flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5" /> Worker 前缀
          </div>
          <div className="text-[11px] text-muted-foreground mt-1 leading-relaxed">
            隔离子进程 mmap 装载融合 KV · prefill ≈ 0（复用已发布前缀）
          </div>
        </div>
      </div>
      {/* KV 流动画轨 */}
      <div className="kv-track h-1.5 rounded-full bg-muted">
        {active && <span className="kv-dot" />}
      </div>
      <div className="space-y-1.5">
        <div className="grid grid-cols-[92px_1fr_150px] items-center gap-2 text-xs">
          <span className="text-right text-muted-foreground text-[11px]">文字通道需发送</span>
          <div className="h-3.5 rounded-full bg-muted overflow-hidden">
            <div className="h-full rounded-full bg-amber-500/80 transition-all duration-1000" style={{ width: active ? '100%' : '8%' }} />
          </div>
          <span className="text-[11px] text-muted-foreground">≈ 笔记全文 + 命令全文</span>
        </div>
        <div className="grid grid-cols-[92px_1fr_150px] items-center gap-2 text-xs">
          <span className="text-right text-muted-foreground text-[11px]">C2C 实际发送</span>
          <div className="h-3.5 rounded-full bg-muted overflow-hidden">
            <div className="h-full rounded-full bg-emerald-500 transition-all duration-1000" style={{ width: active ? '2%' : '0%' }} />
          </div>
          <span className="text-[11px] text-muted-foreground">文字 0 字符 · 仅 KV {kvMb} MB</span>
        </div>
      </div>
      {extra && <p className="text-[11px] text-muted-foreground border-t pt-2">{extra}</p>}
    </div>
  );
}

function StepWorkerTask({ task, active }: { task: any; active: boolean }) {
  const think = useTypewriter(task.think || task.reason || '', active, 55);
  const memPct = Math.min(100, (task.peakIncMib / 512) * 100);
  const maskReport = task.maskReport === true;
  return (
    <div className="space-y-3">
      <div className="rounded-xl border overflow-hidden">
        {[
          ['command', task.command],
          ['deliverable', task.deliverable],
          ['success_criteria', maskAnswer(task.success)],
        ].map(([k, v], i) => (
          <div key={k} className="grid grid-cols-[130px_1fr] border-b last:border-b-0 text-xs">
            <span className="px-3 py-2 text-muted-foreground bg-muted/50 font-mono">{k}</span>
            <span className={`px-3 py-2 font-mono transition-all duration-500 ${active ? 'opacity-100' : 'opacity-40'}`} style={{ transitionDelay: `${i * 150}ms` }}>
              {v}
            </span>
          </div>
        ))}
      </div>
      <div className="border-l-2 border-border pl-3 py-1 text-xs text-muted-foreground bg-muted/30 rounded-r-lg">
        <span className="text-[10px] uppercase tracking-wide">Worker 0.6B · 隔离子进程 · 从公开状态推理</span>
        <p className="mt-1 leading-relaxed min-h-[36px]">{think}</p>
      </div>
      <div className="flex flex-wrap items-center gap-3">
        {maskReport ? (
          <span className="inline-flex items-center gap-1.5 rounded-full bg-muted px-3 py-1 text-xs font-mono text-muted-foreground">
            <Timer className="w-3.5 h-3.5" />
            report = ···（闸门步骤揭晓）
          </span>
        ) : (
          <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 px-3 py-1 text-xs font-mono">
            <CheckCircle2 className="w-3.5 h-3.5" />
            {task.deliverable} = {task.value} ✓ 闸门数值命中
          </span>
        )}
        <span className="inline-flex items-center gap-1 text-[11px] text-muted-foreground font-mono tabular-nums">
          <Zap className="w-3 h-3 text-emerald-500" /> decode {task.decodeS}s
        </span>
        <span className="inline-flex items-center gap-1 text-[11px] text-muted-foreground font-mono tabular-nums">
          <Timer className="w-3 h-3" /> prefill {task.prefillS}s · 输出 {task.outputTokens} tok
        </span>
      </div>
      {/* cgroup memory 计量条 */}
      <div className="grid grid-cols-[92px_1fr_150px] items-center gap-2 text-xs">
        <span className="text-right text-muted-foreground text-[11px]">cgroup 计量</span>
        <div className="h-3.5 rounded-full bg-muted overflow-hidden">
          <div
            className={`h-full rounded-full ${memPct > 80 ? 'bg-red-500' : 'bg-sky-500'} ${active ? 'animate-mem-bar' : 'w-0'} transition-all`}
            style={{ width: active ? `${memPct}%` : '0%' }}
          />
        </div>
        <span className="text-[11px] text-muted-foreground font-mono">
          显存增量 {task.peakIncMib} MiB / 512 MiB 限额
        </span>
      </div>
    </div>
  );
}

function StepFaultGate({ active }: { active: boolean }) {
  const f = DEMO.fault;
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        {[
          ['DONE ×2', '闸门状态'],
          ['2/2', '官方判分 verified'],
          ['2/2', '字段覆盖'],
          [`${f.wallS} s`, '场景墙钟'],
        ].map(([v, l], i) => (
          <div
            key={l}
            className={`rounded-lg border bg-card px-3 py-2 transition-all duration-500 ${active ? 'opacity-100 translate-y-0' : 'opacity-30 translate-y-1'}`}
            style={{ transitionDelay: `${i * 120}ms` }}
          >
            <div className="text-lg font-semibold text-emerald-600 dark:text-emerald-400">{v}</div>
            <div className="text-[11px] text-muted-foreground">{l}</div>
          </div>
        ))}
      </div>
      <div className="rounded-xl border overflow-hidden text-xs">
        <div className="px-3 py-1.5 text-[11px] text-muted-foreground bg-muted/50 flex items-center gap-1.5">
          <GitBranch className="w-3.5 h-3.5" /> 闸门语义核验（公开状态模拟对账）
        </div>
        {[['t1', f.tasks.t1], ['t2', f.tasks.t2]].map(([tid, t]: any) => (
          <div key={tid} className="grid grid-cols-[60px_1fr] border-t">
            <span className="px-3 py-2 text-muted-foreground font-mono">{tid}</span>
            <span className="px-3 py-2 font-mono">{t.gateReason}</span>
          </div>
        ))}
        <div className="grid grid-cols-[60px_1fr] border-t">
          <span className="px-3 py-2 text-muted-foreground font-mono">结论</span>
          <span className="px-3 py-2">内存限额 4096 MB 下 A2 被杀，其余 3 个代理存活</span>
        </div>
      </div>
    </div>
  );
}

/* ---------- 舞台步骤：probe ---------- */
function StepProbePlan({ scenario, active }: { scenario: any; active: boolean }) {
  const typed = useTypewriter(scenario.note || scenario.reason, active, 80);
  return (
    <div className="space-y-3">
      <div className="rounded-xl border-2 border-dashed p-4 bg-muted/40">
        <span className="inline-flex items-center gap-1.5 text-[11px] text-amber-600 dark:text-amber-400 border border-amber-500/40 rounded-full px-2 py-0.5 mb-3">
          <ShieldCheck className="w-3 h-3" />
          Planner 工作笔记（算术脚手架）· 永不以文字发送
        </span>
        <p className="font-mono text-[13px] leading-relaxed min-h-[90px]">{typed}</p>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
        {[
          [scenario.wallS + ' s', 'Planner+臂总墙钟'],
          ['512 tok', '笔记预算（脚手架）'],
          ['8 臂', '同笔记公平对比'],
        ].map(([v, l]) => (
          <div key={l} className="rounded-lg border bg-card px-3 py-2">
            <div className="text-lg font-semibold tabular-nums">{v}</div>
            <div className="text-[11px] text-muted-foreground">{l}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function StepProbeC2c({ scenario, active }: { scenario: any; active: boolean }) {
  const hid = scenario.arms.find((a: any) => a.name === 'c2c_hidden_analysis');
  return (
    <StepC2c
      active={active}
      hiddenTokens={hid?.hiddenTokens || 0}
      kvMb={scenario.kvMb || 0}
      extra={`hidden_analysis 臂：Worker 只读短命令文字，Planner 笔记选区 KV 融合为隐藏前缀（consumed_fused_cache=${hid?.consumedFused ? 'True' : 'False'}）。`}
    />
  );
}

function StepProbeWorker({ scenario, active }: { scenario: any; active: boolean }) {
  const typed = useTypewriter(scenario.thinkMasked || scenario.thinkNote, active, 55);
  const hid = scenario.arms.find((a: any) => a.name === 'c2c_hidden_analysis');
  return (
    <div className="space-y-3">
      <div className="rounded-xl border overflow-hidden">
        {[
          ['command（Worker 唯一可见文字）', scenario.command],
          ['deliverable', `verified value of the pending decision`],
          ['success_criteria', 'result == ···（判分步骤揭晓）'],
        ].map(([k, v], i) => (
          <div key={k} className="grid grid-cols-[200px_1fr] border-b last:border-b-0 text-xs">
            <span className="px-3 py-2 text-muted-foreground bg-muted/50 font-mono">{k}</span>
            <span className={`px-3 py-2 font-mono transition-all duration-500 ${active ? 'opacity-100' : 'opacity-40'}`} style={{ transitionDelay: `${i * 150}ms` }}>
              {v}
            </span>
          </div>
        ))}
      </div>
      <div className="border-l-2 border-border pl-3 py-1 text-xs text-muted-foreground bg-muted/30 rounded-r-lg">
        <span className="text-[10px] uppercase tracking-wide">Worker 0.6B · hidden_analysis 臂推理</span>
        <p className="mt-1 leading-relaxed min-h-[36px]">{typed}</p>
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <span className="inline-flex items-center gap-1.5 rounded-full bg-muted px-3 py-1 text-xs font-mono text-muted-foreground">
          <Timer className="w-3.5 h-3.5" />
          report = ···（判分步骤揭晓）
        </span>
        <span className="inline-flex items-center gap-1 text-[11px] text-muted-foreground font-mono tabular-nums">
          <Zap className="w-3 h-3 text-emerald-500" /> decode {hid?.decodeS}s · 隐藏前缀 {hid?.hiddenTokens} tok
        </span>
      </div>
    </div>
  );
}

function StepProbeArms({ scenario, active }: { scenario: any; active: boolean }) {
  const okArms = scenario.arms.filter((a: any) => a.correct);
  return (
    <div className="space-y-3">
      <div className="text-xs text-muted-foreground">
        同一 Planner 笔记、同一题目，各命中臂的离线判分（expected integer 独立出现在工人报告中即命中）。
        C2C 隐藏融合臂与文字臂同档，即"不靠读 Planner 文字也拿得到背景信息"。
      </div>
      <div className="grid sm:grid-cols-2 gap-1.5">
        {okArms.map((a: any, i: number) => (
          <div
            key={a.name}
            className={`flex items-center justify-between rounded-lg border px-3 py-2 text-xs transition-all duration-500 border-emerald-500/40 bg-emerald-500/5 ${
              active ? 'translate-y-0' : 'translate-y-1 opacity-40'
            }`}
            style={{ transitionDelay: `${i * 60}ms` }}
          >
            <span className="font-mono">{a.label}</span>
            <span className="inline-flex items-center gap-2 font-mono text-emerald-600 dark:text-emerald-400">
              <span className="text-muted-foreground">{a.decodeS}s</span>
              ✓ 命中
            </span>
          </div>
        ))}
      </div>
      <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/5 px-3 py-2 text-[11px] text-muted-foreground flex items-center gap-1.5">
        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400 shrink-0" />
        C2C 隐藏臂（KV 融合、Worker 全程零 Planner 文字）在命中列内，与全部文字臂同档。
      </div>
      <div className="rounded-lg border bg-card px-3 py-2 text-xs flex items-center justify-between">
        <span className="text-muted-foreground">判分揭晓 · expected_answer</span>
        <span className="font-mono font-semibold text-emerald-600 dark:text-emerald-400">= {scenario.expected}</span>
      </div>
    </div>
  );
}

/* ---------- 舞台步骤：multi（多智能体并行巡检） ---------- */
function StepMultiScene({ scenario, active }: { scenario: any; active: boolean }) {
  const tags = ['OS · 内存', 'OS · 磁盘', '非 OS · 库存', '非 OS · 排班'];
  return (
    <div className="space-y-3">
      <p className="text-sm leading-relaxed text-muted-foreground">
        夜班日报发布前，<b className="text-foreground">四块计量表</b>必须全部核验：两块 OS 域（内存限额、SCAN 磁盘调度），
        两块非 OS 域（库存台账、排班合规）。四个值相互独立、各自有界——
        正是一次规划、<b className="text-foreground">四个 Worker 并行委派</b>的理想形状。
      </p>
      <div className="grid sm:grid-cols-2 gap-2">
        {scenario.sceneSteps.map((s: string, i: number) => (
          <div
            key={i}
            className={`rounded-lg border bg-card px-3 py-2.5 transition-all duration-500 ${active ? 'opacity-100 translate-y-0' : 'opacity-30 translate-y-1'}`}
            style={{ transitionDelay: `${i * 120}ms` }}
          >
            <span className={`inline-block rounded-full px-2 py-0.5 text-[10px] font-mono mb-1.5 ${tags[i].startsWith('OS') ? 'bg-sky-500/10 text-sky-600 dark:text-sky-400' : 'bg-violet-500/10 text-violet-600 dark:text-violet-400'}`}>
              {tags[i]}
            </span>
            <p className="text-xs leading-relaxed font-mono">{s}</p>
          </div>
        ))}
      </div>
      <div className="rounded-lg border border-dashed p-3 text-xs text-muted-foreground font-mono">
        pending_decisions × 4 已公开给 Planner；每项验证相互独立、无依赖边 → 可并行下发。
      </div>
    </div>
  );
}

function StepMultiPlan({ scenario, active }: { scenario: any; active: boolean }) {
  const typed = useTypewriter(scenario.note, active, 55);
  return (
    <div className="space-y-3">
      <div className="rounded-xl border-2 border-dashed p-4 bg-muted/40">
        <span className="inline-flex items-center gap-1.5 text-[11px] text-amber-600 dark:text-amber-400 border border-amber-500/40 rounded-full px-2 py-0.5 mb-3">
          <ShieldCheck className="w-3 h-3" />
          Planner 工作笔记 · 永不以文字发送，只以 KV 形式存在
        </span>
        <p className="font-mono text-[13px] leading-relaxed min-h-[60px]">{typed}</p>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        {[
          ['4', '一次委派任务数'],
          ['8', 'max_tasks 行动预算'],
          [scenario.wallS + ' s', 'Planner+三臂总墙钟'],
          [String(scenario.fusion.selectedTokens), '笔记选区 tokens'],
        ].map(([v, l]) => (
          <div key={l} className="rounded-lg border bg-card px-3 py-2">
            <div className="text-lg font-semibold tabular-nums">{v}</div>
            <div className="text-[11px] text-muted-foreground">{l}</div>
          </div>
        ))}
      </div>
      <div className="rounded-xl border overflow-hidden">
        <div className="px-3 py-1.5 text-[11px] text-muted-foreground bg-muted/50 flex items-center gap-1.5">
          <GitBranch className="w-3.5 h-3.5" /> 一份行动 JSON · tasks 数组四项并行委派
        </div>
        {scenario.tasks.map((t: any, i: number) => (
          <div key={t.id} className="grid grid-cols-[52px_1fr] border-t text-xs">
            <span className="px-3 py-2 text-emerald-600 dark:text-emerald-400 font-mono bg-muted/30">{t.id}</span>
            <span className={`px-3 py-2 font-mono transition-all duration-500 ${active ? 'opacity-100' : 'opacity-40'}`} style={{ transitionDelay: `${i * 150}ms` }}>
              {maskAnswer(t.command)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function StepMultiWorkers({ scenario, active }: { scenario: any; active: boolean }) {
  const arm = scenario.arms.find((a: any) => a.name === 'c2c_hidden_analysis');
  return (
    <div className="space-y-3">
      <div className="text-xs text-muted-foreground">
        四个 Worker 复用同一融合前缀并行解码（c2c_hidden_analysis 臂：Worker 全程零 Planner 文字，
        只读短命令 + 笔记 KV 隐藏融合）。命令中的核算结果已脱敏，报告数值在判分步骤统一揭晓。
      </div>
      <div className="grid sm:grid-cols-2 gap-2">
        {scenario.tasks.map((t: any, i: number) => {
          const call = arm.calls[i];
          return (
            <div
              key={t.id}
              className={`rounded-lg border bg-card p-3 transition-all duration-500 ${active ? 'opacity-100 translate-y-0' : 'opacity-30 translate-y-1'}`}
              style={{ transitionDelay: `${i * 130}ms` }}
            >
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs font-semibold font-mono text-emerald-600 dark:text-emerald-400">{t.id} · 0.6B Worker</span>
                <span className="text-[10px] text-muted-foreground font-mono tabular-nums">decode {call.decodeS}s</span>
              </div>
              <p className="text-[11px] font-mono text-muted-foreground leading-relaxed mb-1.5">{maskAnswer(t.command)}</p>
              <div className="rounded bg-muted/50 px-2 py-1.5 text-[11px] font-mono text-muted-foreground">
                report = ···（判分步骤揭晓）
              </div>
            </div>
          );
        })}
      </div>
      <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/5 px-3 py-2 text-[11px] text-muted-foreground flex items-center gap-1.5">
        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400 shrink-0" />
        一次融合 {scenario.fusion.sharedFusionSeconds}s 完成，四工人各自复用已发布前缀（fusion_reused_by_each_worker=True），缓存分支私有。
      </div>
    </div>
  );
}

function StepMultiGate({ scenario, active }: { scenario: any; active: boolean }) {
  return (
    <div className="space-y-3">
      <div className={`rounded-lg border bg-card px-3 py-2 text-xs flex items-center justify-between transition-all duration-700 ${active ? 'opacity-100' : 'opacity-40'}`}>
        <span className="text-muted-foreground">判分揭晓 · expected_answers</span>
        <span className="font-mono font-semibold text-emerald-600 dark:text-emerald-400">= [12152, 543, 300, 109]</span>
      </div>
      <div className="rounded-xl border overflow-hidden text-xs">
        <div className="px-3 py-1.5 text-[11px] text-muted-foreground bg-muted/50 flex items-center gap-1.5">
          <GitBranch className="w-3.5 h-3.5" /> 离线判分矩阵（工人报告中的独立整数 == 期望值即命中）
        </div>
        <div className="grid grid-cols-[150px_repeat(4,1fr)_44px] border-t text-[11px] font-mono">
          <span className="px-2 py-1.5 text-muted-foreground bg-muted/30">臂 \ 工人</span>
          {['W1', 'W2', 'W3', 'W4'].map((w) => (
            <span key={w} className="px-2 py-1.5 text-center text-muted-foreground bg-muted/30">{w}</span>
          ))}
          <span className="px-2 py-1.5 text-center text-muted-foreground bg-muted/30">命中</span>
        </div>
        {scenario.arms.map((a: any) => (
          <div key={a.name} className="grid grid-cols-[150px_repeat(4,1fr)_44px] border-t font-mono">
            <span className="px-2 py-1.5 text-muted-foreground truncate" title={a.label}>{a.name}</span>
            {a.calls.map((c: any, i: number) => (
              <span key={i} className={`px-2 py-1.5 text-center ${c.correct ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-500'}`}>
                {c.correct ? '✓' : '✗'}
              </span>
            ))}
            <span className="px-2 py-1.5 text-center font-semibold">{a.hits}</span>
          </div>
        ))}
      </div>
      <div className="rounded-lg border px-3 py-2 text-[11px] text-muted-foreground leading-relaxed">
        闸门逐值对账：W2 / W4 五臂全中；W1 的读命令臂全部照抄 Planner 命令中的算术笔误（12142 ≠ 12152）被闸门拒绝，
        而 state_native 从公开状态独立重算得 12152 ✓——闸门 + 公开状态对账拦下了 Planner 自身的错误；
        W3 的 c2c_shared 未报数、hidden_analysis 解码退化，判无效入有界修复队列（修复→重派→再核验闭环已在 acbi4 批次实证）。
      </div>
    </div>
  );
}

/* ---------- 步骤定义 ---------- */
interface StepDef {
  id: string;
  no: string;
  title: string;
  durMs: number;
}

const FAULT_STEPS: StepDef[] = [
  { id: 'scene', no: '0', title: '场景装载：内存限额记账', durMs: 5000 },
  { id: 'plan', no: '1', title: 'Planner 规划（笔记只以 KV 存在）', durMs: 9000 },
  { id: 'c2c', no: '2', title: 'C2C 交接 + mmap 装载', durMs: 8000 },
  { id: 't1', no: '3', title: 'Worker t1 · 子进程执行', durMs: 7000 },
  { id: 't2', no: '4', title: 'Worker t2 · LRU 池置换', durMs: 7000 },
  { id: 'gate', no: '5', title: '闸门核验与汇总', durMs: 6000 },
];

const PROBE_STEPS: StepDef[] = [
  { id: 'scene', no: '0', title: '场景装载', durMs: 4500 },
  { id: 'plan', no: '1', title: 'Planner 规划（算术脚手架）', durMs: 9000 },
  { id: 'c2c', no: '2', title: 'C2C 交接：KV 选区融合', durMs: 7000 },
  { id: 'worker', no: '3', title: 'Worker 执行（只读短命令）', durMs: 7000 },
  { id: 'arms', no: '4', title: '离线判分对比 · 命中臂', durMs: 6000 },
];

const MULTI_STEPS: StepDef[] = [
  { id: 'scene', no: '0', title: '四表巡检装载', durMs: 5000 },
  { id: 'plan', no: '1', title: 'Planner 一次规划 4 项委派', durMs: 8000 },
  { id: 'c2c', no: '2', title: 'C2C 一次融合 · 四工人复用', durMs: 7000 },
  { id: 'workers', no: '3', title: 'W1–W4 并行执行', durMs: 7000 },
  { id: 'gate', no: '4', title: '闸门判分与修复入队', durMs: 6000 },
];

const SCENARIOS = [
  { key: 'fault', label: '内存限额记账 · 全设计版' },
  { key: 'deadlock', label: '死锁检测' },
  { key: 'pagetable', label: '页表地址翻译' },
  { key: 'ratelimit', label: '令牌桶限流' },
  { key: 'multi6', label: '多智能体并行巡检' },
] as const;

/* ---------- App ---------- */
export default function App() {
  const [scenKey, setScenKey] = useState<string>('fault');
  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(true);
  const stepIdxRef = useRef(stepIdx);
  stepIdxRef.current = stepIdx;
  const playingRef = useRef(playing);
  playingRef.current = playing;

  const isFault = scenKey === 'fault';
  const isMulti = scenKey === 'multi6';
  const steps = isFault ? FAULT_STEPS : isMulti ? MULTI_STEPS : PROBE_STEPS;
  const scenario: any = (DEMO as any)[scenKey];

  useEffect(() => {
    setStepIdx(0);
    setPlaying(true);
  }, [scenKey]);

  useEffect(() => {
    if (!playing) return;
    const t = setInterval(() => {
      const cur = stepIdxRef.current;
      if (cur >= steps.length - 1) {
        setPlaying(false);
        clearInterval(t);
        return;
      }
      setStepIdx(cur + 1);
    }, steps[Math.min(stepIdxRef.current, steps.length - 1)].durMs);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, scenKey]);

  const step = steps[stepIdx];

  const stage = useMemo(() => {
    if (isFault) {
      const f = DEMO.fault;
      switch (step.id) {
        case 'scene':
          return <StepFaultScene />;
        case 'plan':
          return (
            <StepPlan
              active
              note={f.reasonMasked || f.note || f.reason}
              stats={[
                [String(f.plannerTokens ?? 363), '规划生成 tokens'],
                [`${f.kvMb} MB`, '随融合搬运的 KV'],
                [`${f.plannerWallS} s`, 'Planner 生成墙钟'],
                ['8', 'max_tasks 行动预算'],
              ]}
            />
          );
        case 'c2c':
          return (
            <StepC2c
              active
              hiddenTokens={f.tasks.t1.hiddenTokens}
              kvMb={f.kvMb}
              extra="融合后的前缀以 .cache.bin mmap 发布进 LRU 池（容量 1）；子进程启动时映射装载，host 侧零拷贝。cgroup 控制器开始计量 tokens/memory。"
            />
          );
        case 't1':
          return <StepWorkerTask task={{ ...DEMO.fault.tasks.t1, maskReport: true, think: 'A2 退出码 137、峰值 8200 MB 超限 → 超限代理序号由 Worker 从公开状态算出（闸门步骤揭晓）；resource_control 记账：输出 323 tokens、峰值 191.5 MiB（worker 限额 24576 MiB 内），零违规。' }} active />;
        case 't2':
          return <StepWorkerTask task={{ ...DEMO.fault.tasks.t2, maskReport: true, think: '退出码 0 的代理存活计数由 Worker 算出（闸门步骤揭晓）。装载 t2 载荷时 LRU 池逐出 t1 载荷（swap-out 事件在案），t2 映射装载产生一次缺页式重载。' }} active />;
        case 'gate':
          return <StepFaultGate active />;
        default:
          return null;
      }
    }
    if (isMulti) {
      switch (step.id) {
        case 'scene':
          return <StepMultiScene scenario={scenario} active />;
        case 'plan':
          return <StepMultiPlan scenario={scenario} active />;
        case 'c2c':
          return (
            <StepC2c
              active
              hiddenTokens={scenario.fusion.selectedTokens}
              kvMb={scenario.kvMb || 0}
              extra={`一次融合 ${scenario.fusion.sharedFusionSeconds}s 完成并发布共享缓存；4 个 Worker 各自复用同一融合前缀（fusion_reused_by_each_worker=True），缓存分支私有，文字通道发送 0 字符。`}
            />
          );
        case 'workers':
          return <StepMultiWorkers scenario={scenario} active />;
        case 'gate':
          return <StepMultiGate scenario={scenario} active />;
        default:
          return null;
      }
    }
    switch (step.id) {
      case 'scene':
        return (
          <div className="space-y-3">
            <p className="text-sm leading-relaxed text-muted-foreground">{scenario.question}</p>
            <div className="rounded-lg border border-dashed p-3 text-xs text-muted-foreground font-mono">
              pending_decision 已公开给 Planner；验证该值就是本次有用的委派。
            </div>
          </div>
        );
      case 'plan':
        return <StepProbePlan scenario={scenario} active />;
      case 'c2c':
        return <StepProbeC2c scenario={scenario} active />;
      case 'worker':
        return <StepProbeWorker scenario={scenario} active />;
      case 'arms':
        return <StepProbeArms scenario={scenario} active />;
      default:
        return null;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step.id, scenKey]);

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-5xl mx-auto px-5 py-6 space-y-5">
        <header className="space-y-2">
          <div className="flex items-center gap-2 text-[11px] text-muted-foreground flex-wrap">
            <HardDrive className="w-3.5 h-3.5" />
            C2C 多智能体运行时 · 真实场景实例集 · Qwen3-4B Planner → Qwen3-0.6B Worker · seed 42
          </div>
          <h1 className="text-xl font-semibold">OS 场景委派实例 · 五个真实运行</h1>
          <div className="flex flex-wrap gap-1.5">
            {SCENARIOS.map((s) => (
              <button
                key={s.key}
                onClick={() => setScenKey(s.key)}
                className={`rounded-full border px-3 py-1.5 text-xs transition-colors ${
                  scenKey === s.key ? 'bg-foreground text-background border-foreground' : 'border-border text-muted-foreground hover:bg-muted'
                }`}
              >
                {s.label}
              </button>
            ))}
          </div>
          <p className="text-xs text-muted-foreground">{scenario.subtitle}</p>
        </header>

        {/* 步骤条 */}
        <div className="flex flex-wrap gap-1.5">
          {steps.map((s, i) => (
            <button
              key={s.id}
              onClick={() => {
                setStepIdx(i);
                setPlaying(false);
              }}
              className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs transition-colors ${
                i === stepIdx
                  ? 'bg-foreground text-background border-foreground'
                  : i < stepIdx
                    ? 'border-emerald-500/50 text-emerald-600 dark:text-emerald-400'
                    : 'border-border text-muted-foreground hover:bg-muted'
              }`}
            >
              <span
                className={`w-4 h-4 rounded-full text-[10px] inline-flex items-center justify-center ${
                  i === stepIdx
                    ? 'bg-emerald-500 text-white'
                    : i < stepIdx
                      ? 'bg-emerald-500/20 text-emerald-600 dark:text-emerald-400'
                      : 'bg-muted text-muted-foreground'
                }`}
              >
                {i < stepIdx ? '✓' : s.no}
              </span>
              {s.title}
            </button>
          ))}
        </div>

        {/* 主体 */}
        <div className="grid md:grid-cols-[300px_1fr] gap-4 items-start">
          {isFault ? <FaultPanel stepIdx={stepIdx} steps={FAULT_STEPS} /> : <ProbePanel scenario={scenario} stepIdx={stepIdx} />}
          <div className="rounded-xl border bg-card p-4 min-h-[340px]">
            <div className="text-xs text-muted-foreground mb-3 font-mono">
              步骤 {step.no} / {steps.length - 1} · {step.title}
            </div>
            {stage}
          </div>
        </div>

        {isFault && <EventLog stepIdx={stepIdx} total={FAULT_STEPS.length} />}

        {/* 控制条 */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => setStepIdx(0)}
            className="inline-flex items-center gap-1 rounded-lg border px-3 py-1.5 text-xs hover:bg-muted"
          >
            <SkipBack className="w-3.5 h-3.5" /> 重播
          </button>
          <button
            onClick={() => {
              if (stepIdx >= steps.length - 1) setStepIdx(0);
              setPlaying((p) => !p);
            }}
            className="inline-flex items-center gap-1 rounded-lg bg-foreground text-background px-4 py-1.5 text-xs"
          >
            {playing ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
            {playing ? '暂停' : '自动播放'}
          </button>
          <button
            onClick={() => {
              setPlaying(false);
              setStepIdx((s) => Math.min(s + 1, steps.length - 1));
            }}
            className="inline-flex items-center gap-1 rounded-lg border px-3 py-1.5 text-xs hover:bg-muted"
          >
            <SkipForward className="w-3.5 h-3.5" /> 下一步
          </button>
          <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
            <div className="h-full bg-emerald-500 transition-all duration-700" style={{ width: `${((stepIdx + 1) / steps.length) * 100}%` }} />
          </div>
          <span className="text-[11px] text-muted-foreground font-mono tabular-nums">
            {stepIdx + 1}/{steps.length}
          </span>
        </div>

        <footer className="text-[11px] text-muted-foreground border-t pt-3 leading-relaxed">
          数据来源：E:\code\c2cos\artifacts\probe4-batch-20261020（acbi4 隔离全设计批、probe4/probe5 场景批，2026-10-20 晚真实运行）、
          E:\code\c2cos\artifacts\probe6-batch-20261021（probe6 多智能体并行巡检批，2026-10-21 晚真实运行）。
          结果文档：docs\probe4-probe5-acbi4-results-2026-10-20.md。
        </footer>
      </div>
    </div>
  );
}
