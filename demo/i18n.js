/* Presentation-only translations. Raw run data, identifiers and results are unchanged. */
(function () {
  'use strict';
  let lang = new URLSearchParams(location.search).get('lang') === 'en' ? 'en' : 'zh';
  const originals = new WeakMap();
  const pairs = [
    ['系统综合','Integrated runtime'],
    ['一次任务失败，运行时如何继续处理','How the runtime handles a failed task'],
    ['同一实例贯通 ACB、资源准入、隔离进程、mmap、载荷管理、验收与有界修复。','One run connects ACBs, admission, isolated processes, mmap, payload management, validation and bounded repair.'],
    ['同一次历史 GPU 运行 · acbi-osclassics-sched-poolmmap','One recorded GPU run · acbi-osclassics-sched-poolmmap'],
    ['ACB 生命周期','ACB lifecycle'],['GPU 预算准入','GPU budget admission'],['独立进程与 RSS 监督','Isolated processes and RSS supervision'],['mmap 载荷装载','mmap payload loading'],['验收反馈与有界修复','Validation feedback and bounded repair'],['载荷换出与文件恢复','Payload eviction and file restore'],
    ['一个场景：四作业 SJF 指标核验','One scenario: validate SJF metrics for four jobs'],
    ['t1：总周转时间；t2：总等待时间。Planner 实际派出两个独立任务。','t1: total turnaround; t2: total waiting. The Planner dispatched two independent tasks.'],
    ['当前阶段','Current stage'],['准备回放','Ready to replay'],
    ['装载公开作业','Load public jobs'],['建立 ACB 并准入','Create ACBs and admit'],['t1 拒绝并创建修复','Reject t1 and create repair'],['t2 拒绝并创建修复','Reject t2 and create repair'],['t1.r1 修复通过','Accept t1.r1 repair'],['t2.r1 仍失败','t2.r1 still fails'],['停止重试并保留失败','Stop retrying and preserve failure'],['汇总全部运行证据','Summarize run evidence'],
    ['等待隔离执行','Await isolated execution'],['实际修复反馈','Actual repair feedback'],
    ['资源与隔离','Resources and isolation'],['载荷池与 mmap','Payload pool and mmap'],['等待执行记录','Await execution record'],['等待载荷事件','Await payload events'],
    ['恢复结果与边界','Recovery outcome and scope'],
    ['4 次尝试，2 个修复任务，1 次验收通过；t1 修复成功，t2 修复后仍失败，不再重试。','4 attempts, 2 repair tasks, 1 accepted attempt. t1 recovers; t2 still fails after repair and is not retried.'],
    ['4 个不同 PID，均正常退出；本例没有 OOM 或超时。全程墙钟 91.6 s。','4 distinct PIDs, all exiting normally; no OOM or timeout in this run. Whole-run wall time: 91.6 s.'],
    ['原始 depends_on 为空；图中虚线是修复来源关系。本例未触发依赖等待、CFS 对照或银行家算法。','Original depends_on lists are empty; dashed arrows show repair lineage. This run does not exercise dependency blocking, CFS comparisons or banker admission.'],
    ['换回为应用文件事件，不代表内核缺页，也不代表旧载荷被实际复用。','Restores are application file events, not kernel page faults or evidence of actual old-payload reuse.'],
    ['四次尝试对应 tasks/t1.json、t2.json、t1.r1.json、t2.r1.json。','The four attempts are recorded in tasks/t1.json, t2.json, t1.r1.json and t2.r1.json.'],
    ['历史真实运行回放 · 非通信机制贯通','Recorded run replay · integrated non-communication mechanisms'],
    ['上次尝试被拒绝：sjf_total_turnaround_seconds 报告0，公开状态模拟结果为20。请根据公开证据重新计算，只报告纠正后的结果。','A previous attempt of this task was rejected (sjf_total_turnaround_seconds: worker reported 0, public-state simulation says 20). Recompute the value from the public evidence and report only the corrected result.'],
    ['上次尝试被拒绝：sjf_total_waiting_seconds 报告5，公开状态模拟结果为10。请根据公开证据重新计算，只报告纠正后的结果。','A previous attempt of this task was rejected (sjf_total_waiting_seconds: worker reported 5, public-state simulation says 10). Recompute the value from the public evidence and report only the corrected result.'],
    ['OS 调度','OS scheduling'],
    ['同一批任务，调度顺序影响完成体验','Same tasks, different scheduling outcomes'],
    ['从就绪队列到 ACB 状态：对比 FIFO 与服务记账，观察短任务如何被提前执行。','From the ready queue to ACB states: compare FIFO with service accounting and see short tasks run earlier.'],
    ['历史调度组件验证 · 确定性负载 · 非 GPU 推理','Recorded scheduler component check · deterministic workload · no GPU inference'],
    ['相同输入','Identical input'],
    ['big1 / big2：各 400 ms；s1–s6：各 30 ms','big1 / big2: 400 ms each; s1–s6: 30 ms each'],
    ['8 个任务同时就绪；总服务量 980 ms','8 tasks ready together; total service: 980 ms'],
    ['FIFO · 按提交顺序','FIFO · submission order'],
    ['服务记账 · 同值时短任务优先','Service accounting · shorter tasks break ties'],
    ['ACB 状态','ACB states'],
    ['每一步分别回放两种策略的一次任务完成；两侧不代表同一墙钟时刻。','Each step replays one completion per policy; the two lanes are not synchronized wall-clock snapshots.'],
    ['平均周转时间','Mean turnaround'],
    ['相同服务量，改变调度顺序，减少短任务被长任务阻塞。','Same total service, reordered execution: short tasks spend less time behind long tasks.'],
    ['验证依据','Evidence'],
    ['checks.json：11 项组件检查通过；本页使用其中的调度顺序、平均周转时间与 I1/I2 验证结果。时间线按测试脚本中的确定性服务时长重建。','checks.json: all 11 component checks passed. This page uses the recorded order, turnaround and I1/I2 checks. Timelines are reconstructed from deterministic service durations in the test script.'],
    ['历史调度组件验证 · 2026-10-19','Recorded scheduler component check · 2026-10-19'],
    ['委派运行','Delegation replay'],['五方案对比','Five-scheme comparison'],
    ['一次真实的 C2C 委派','A real C2C delegation'],['五个通信方案，同一道题','Five communication schemes, one task'],
    ['Planner 只发','The Planner sends only '],['短命令','short commands'],['；语义走','; semantics travel through '],['共享 KV','shared KV'],['——一次融合，两个 Worker 复用。',' — one fusion, reused by two Workers.'],
    ['同一份公开状态、同一套判分。隐藏方案只读','Same public state and grading. Hidden schemes read only '],['，语义由','; semantics arrive via '],['融合 KV','fused KV'],['送达；五个方案全部通过闸门。','; all five schemes pass the gate.'],
    ['c2c 隐藏方案只读短命令；语义由融合 KV 送达。','Hidden C2C schemes read short commands; semantics arrive via fused KV.'],
    ['死锁检测','Deadlock detection'],['页表地址翻译','Page-table translation'],['令牌桶限流','Token-bucket admission'],
    ['短命令 + analysis KV','Short command + analysis KV'],['短命令 + 全转录 KV','Short command + full-transcript KV'],['全转录文字 + 融合 KV','Full transcript + fused KV'],['4B 简报文字','4B-generated handoff'],['全转录文字','Full transcript text'],
    ['1 次融合 · 共享给 2 个 Worker','1 fusion · shared by 2 Workers'],['同一份共享前缀 · 2 条私有分支','One shared prefix · 2 private branches'],
    ['共享给 2 个 Worker · 分支私有','Shared by 2 Workers · private branches'],['1 次融合 → 2 个 Worker','1 fusion → 2 Workers'],
    ['方案 / 可见输入','Scheme / visible input'],['输出','Output'],['判定','Verdict'],['融合','Fused'],['✓ 通过','✓ Passed'],
    ['播放','Play'],['暂停','Pause'],['重播','Replay'],['上一步','Previous step'],['下一步','Next step'],
    ['真实 GPU 运行 · acbi4 · 2026-10-20','Recorded GPU run · acbi4 · 2026-10-20'],
    ['KV 通道','KV channel'],['隐藏方案使用','Used by hidden schemes'],['Worker 只读这一句','The Worker reads this short command'],
    ['P0→P1→P2→P0 构成环，P3→P1 不构成环。环上不同进程数由 Worker 独立报出。','P0→P1→P2→P0 forms a cycle; P3→P1 is outside it. The Worker reports the number of distinct processes in the cycle.'],
    ['帧 8 × 4096 = 32768，再加页内偏移得物理地址。最终数值由 Worker 独立报出。','Frame 8 × 4096 = 32768; add the page offset to obtain the physical address. The Worker reports the final value.'],
    ['t=0 准入余 7；t=1 准入余 6；t=2 准入余 3；t=3 桶内不足、丢弃。准入请求数由 Worker 独立报出。','At t=0, admit and leave 7; t=1, admit and leave 6; t=2, admit and leave 3; t=3, reject for insufficient tokens. The Worker reports the admitted count.'],
    ['规划器需要确定哪个代理因内存限额被终止，以及有多少代理正常完成。这些值是控制器推进计划所必需的。内存限额为4096 MB，被终止的代理A2峰值RSS为8200 MB，超过限额。其余代理A1、A3、A4的退出码为0，均正常完成。Worker根据公开状态计算两项结果，再由闸门核验。','The planner needs to determine which agent was killed by the memory limit and how many agents completed normally. These values are critical for the controller to proceed with the plan. The memory limit is 4096 MB, and the peak RSS of the killed agent (A2) is 8200 MB, which exceeds the limit. The other agents (A1, A3, A4) completed normally with exit code 0. The two indices themselves are computed by the Workers from this public state and verified by the gate.'],
    ['计算退出码为137且peak_rss_mb > 4096的代理序号，作为oom_agent_index','compute oom_agent_index as the index of the agent with exit code 137 and peak_rss_mb > 4096'],
    ['统计退出码为0的代理数量，作为survived_siblings','compute survived_siblings as the count of agents with exit code 0'],
    ['等待边为P0→P1、P1→P2、P2→P0、P3→P1。共有多少个不同进程处于等待环中？','The wait-for edges are P0->P1, P1->P2, P2->P0, P3->P1. How many distinct processes participate in a wait-for cycle?'],
    ['根据等待边P0→P1、P1→P2、P2→P0、P3→P1，返回环内不同进程的数量','Return the number of distinct processes in the wait-for cycle based on the reported edges: P0->P1, P1->P2, P2->P0, P3->P1'],
    ['页表项有效，将虚拟帧8映射到物理帧8，页大小4096 B。虚拟地址34083的页内偏移为1315，其物理地址是多少字节？','A present page map entry maps virtual frame 8 to physical frame 8 with page size 4096 B. For virtual address 34083 (offset 1315 within its page), what is the physical address in bytes?'],
    ['根据给定公开状态计算physical_address_bytes','compute physical_address_bytes from the granted public state'],
    ['令牌桶容量10，每秒补充1个令牌；t=0、1、2、3的请求大小分别为3、2、4、5。审计窗口内4个请求中有多少被准入？','How many of the 4 requests in the audit window are admitted by the token bucket (capacity 10, refill 1 token/s, requests at t=0,1,2,3 with sizes 3,2,4,5)?'],
    ['根据令牌桶算法，返回审计窗口内获准请求的数量。','Return the number of admitted requests in the audit window based on the token bucket algorithm.'],
    ['物理地址为34083字节。','The physical address in bytes is 34083.']
  ];
  const terms = [
    ['等待修复','REPAIR_PENDING'],['失败','FAILED'],['等待模型','WAIT_MODEL'],['到达','arrival'],['运行时长','burst'],['报告','Report'],
    ['任务显存估算','Estimated GPU memory per task'],['准入预算','admission budget'],['RSS 上限','RSS limit'],['超时','timeout'],['当前记录：','Current record: '],['容量','Capacity'],['；每次尝试独立发布并映射载荷。','; each attempt publishes and maps its own payload.'],['同一次运行','One recorded run'],['个独立任务','independent tasks'],['尝试','Attempts'],['修复','Repairs'],
    ['已消耗服务时间：','Service consumed: '],['已完成短任务：','Short tasks completed: '],['服务记账','Service accounting'],['已回放','Replayed'],['任务','Tasks'],['载入历史调度记录','Load scheduler evidence'],['个任务','tasks'],['通过','passed'],['已完成','completed'],
    ['已完成','DONE'],['等待中','QUEUED'],['运行中','RUNNING'],['就绪','READY'],['已通过','ACCEPTED'],['待核验','PENDING'],['全部完成','COMPLETE'],
    ['运行日志','RUN LOG'],['真实运行','REAL RUN'],['五种方案 · 同一任务','FIVE SCHEMES · SAME TASK'],
    ['规划器','Planner'],['执行器','Worker'],['融合器','C2C fuser'],['公开状态验收','Public-state gate'],['公开状态闸门','public-state gate'],
    ['投影 → 接收模型前缀','projection → receiver prefix'],['预期','expected'],['实际报告','reported'],['已接受','accepted'],
    ['总墙钟','WALL'],['规划','PLANNER'],['隐藏选区','HIDDEN'],['方案','SCHEMES'],['公开状态','PUBLIC STATE'],['短命令 · 规划器 → 执行器','SHORT COMMAND · PLANNER → WORKER'],['短命令','SHORT COMMAND'],['共享KV','SHARED KV'],['验收','GATE'],
    ['运行已载入','run loaded'],['公开状态','public state'],['内存阈值','memory_max'],['分析token','analysis tokens'],['token','tokens'],
    ['任务预算','action budget'],['委派动作','action delegate'],['个任务','tasks'],['个代理','agents'],['融合','fuse'],['发布','publish'],['由以下执行器共享','shared by'],['执行器','workers'],
    ['解码','decode'],['完成','done'],['运行中','running'],['委派','delegate'],['共享KV','shared KV'],['规划器','planner'],['验收','gate'],['已接受','accepted'],['预填','prefill'],['运行完成','run completed'],['不变量','invariants'],['不变量通过','invariants ok'],['墙钟','wall'],
    ['五方案对比','five-scheme comparison'],['五方案均通过','all five schemes accepted'],['公开状态模拟','public-state simulation'],['同一任务 · 五种方案','same task · five schemes'],['固定融合器','frozen fuser'],
    ['次融合','次融合'],['方案通过公开状态闸门','schemes passed the public-state gate'],['方案通过','schemes passed'],['次融合','fusion']
  ];
  // Exact phrases first, then small UI labels. Model/JSON identifiers remain intact.
  const exact = {zh:new Map(),en:new Map()};
  pairs.forEach(([zh,en])=>{exact.zh.set(en,zh);exact.en.set(zh,en);});
  const skip = n => n.parentElement && n.parentElement.closest('script,style,[data-i18n-ignore],.report,.action');
  function translated(raw) {
    const trimmed=raw.trim(); if (!trimmed) return raw;
    const found=exact[lang].get(trimmed);
    if(found!==undefined) return raw.replace(trimmed,found);
    let s=raw;
    const list=terms.map(([zh,en])=>lang==='zh'?[en,zh]:[zh,en]).filter(([a,b])=>a!==b).sort((a,b)=>b[0].length-a[0].length);
    // One-pass replacement avoids translating newly inserted terms again.
    const lookup=new Map(list);const escaped=list.map(([a])=>{
      const q=a.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
      return /^[A-Za-z]/.test(a)?'(?<![A-Za-z0-9_])'+q+'(?![A-Za-z0-9_])':q;
    });
    if(escaped.length)s=s.replace(new RegExp(escaped.join('|'),'g'),m=>lookup.get(m));
    return s;
  }
  let pending=false;
  const observer=new MutationObserver(()=>{if(!pending){pending=true;queueMicrotask(()=>{pending=false;refresh();});}});
  function refresh(){
    observer.disconnect();
    const walker=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT);
    while(walker.nextNode()){
      const n=walker.currentNode;if(skip(n))continue;
      let rec=originals.get(n);if(!rec||n.nodeValue!==rec.shown)rec={raw:n.nodeValue,shown:n.nodeValue};
      const next=translated(rec.raw);if(n.nodeValue!==next)n.nodeValue=next;rec.shown=next;originals.set(n,rec);
    }
    document.documentElement.lang=lang==='zh'?'zh-CN':'en';
    document.querySelectorAll('[data-language]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.language===lang)));
    document.getElementById('prev').setAttribute('aria-label',lang==='zh'?'上一步':'Previous step');
    document.getElementById('next').setAttribute('aria-label',lang==='zh'?'下一步':'Next step');
    document.title=lang==='zh'?'C2C · 委派运行':'C2C · Delegation replay';
    observer.observe(document.body,{subtree:true,childList:true,characterData:true});
  }
  document.querySelectorAll('[data-language]').forEach(b=>b.addEventListener('click',()=>{lang=b.dataset.language;refresh();}));
  // Keep the brand link inside this file, including file:// offline use.
  const brand=document.querySelector('.brand');if(brand)brand.setAttribute('href',location.pathname);
  refresh();
})();
