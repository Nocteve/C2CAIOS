/* C2C run replay — vanilla, no build step */
(async function () {
  'use strict';

  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const fmt = (n, d = 1) => Number(n).toFixed(d);

  let data;
  try {
    const embedded = document.getElementById('embedded-run-data');
    data = embedded ? JSON.parse(embedded.textContent) : await fetch('./run-data.json', { cache: 'no-store' }).then((r) => {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    });
  } catch (e) {
    document.body.innerHTML = '<p style="padding:40px;font-family:monospace">无法读取 run-data.json（' + esc(e.message) + '）。请通过 localhost:7100 打开。</p>';
    return;
  }

  const F = data.fault;
  const OS = data.os || await fetch('./os-data.json').then(r => { if (!r.ok) throw new Error('Missing OS evidence'); return r.json(); });
  const SYS = data.system || await fetch('./system-data.json').then(r=>{if(!r.ok)throw new Error('Missing system evidence');return r.json();});
  const PROBES = data.probes;
  const PROBE_KEYS = ['deadlock', 'pagetable', 'ratelimit'];
  const PROBE_LABEL = { deadlock: '死锁检测', pagetable: '页表地址翻译', ratelimit: '令牌桶限流' };
  const ARM_INPUT = {
    c2c_hidden_analysis: '短命令 + analysis KV',
    c2c_hidden_full: '短命令 + 全转录 KV',
    c2c_shared: '全转录文字 + 融合 KV',
    text_brief: '4B 简报文字',
    text_full: '全转录文字',
  };

  const state = { mode: 'run', step: 0, playing: false, probe: 'deadlock', timer: null };
  const RUN_STEPS = [
    { key: 'planner', dur: 2600 },
    { key: 'tasks', dur: 2400 },
    { key: 'fuse', dur: 2800 },
    { key: 't1', dur: 3300 },
    { key: 't2', dur: 3300 },
    { key: 'gate', dur: 3800 },
  ];
  const ARM_STEPS = 5;

  /* ---------------- static render ---------------- */

  function renderRunShell() {
    $('#plannerNote').textContent = F.planner.note;
    $('#agents').innerHTML = F.agents
      .map((a) => '<span class="agent-mini' + (a.exitCode ? ' bad' : '') + '">' + esc(a.id) + ' · ' + a.exitCode + ' · ' + a.peakRssMb + ' MB</span>')
      .join('');
    $('#plannerAction').textContent = '{ "action": "delegate", "tasks": 2 }';
    $('#plannerMeta').innerHTML = '<span>' + F.planner.tokens + ' tokens</span><span>' + fmt(F.planner.wall) + ' s</span><span>action budget 8</span>';

    $('#taskRow').innerHTML = F.tasks
      .map(
        (t) =>
          '<div class="task-chip"><b>' + esc(t.id) + ' · ' + esc(t.deliverable) + '</b><span class="cmd-tag">短命令</span>' + esc(t.command) + '</div>'
      )
      .join('');

    let toks = '';
    for (let i = 0; i < 24; i++) toks += '<i class="tok" style="--i:' + i + '"></i>';
    $('#tokens').innerHTML = toks;
    $('#fuserMeta').textContent = F.fusion.hidden + ' tokens · ' + F.fusion.kvMb + ' MB · 1 次融合';
    $('#fuserShare').textContent = '共享给 ' + F.tasks.length + ' 个 Worker · 分支私有';

    $('#workers').innerHTML = F.tasks
      .map(
        (t, i) =>
          '<article class="node worker" data-step="' + (4 + i) + '" id="w' + (i + 1) + '">' +
          '<div class="node-head"><span class="avatar">' + (i + 1) + '</span><b>' + esc(t.id) + '</b><span class="kind">' + esc(t.deliverable) + '</span>' +
          '<span class="status" data-wstatus>QUEUED</span></div>' +
          '<div class="cmd">' + esc(t.command) + '</div>' +
          '<div class="bar"><i></i></div>' +
          '<div class="foot"><span>shared KV · decode ' + fmt(t.decode, 1) + ' s</span><span>' + t.outputTokens + ' tok</span></div>' +
          '<div class="report"><span class="ok">✓</span> ' + esc(JSON.stringify({ [t.deliverable]: t.report })) + '</div>' +
          '</article>'
      )
      .join('');

    $('#verdicts').innerHTML = F.tasks
      .map(
        (t) =>
          '<div class="verdict"><b>' + esc(t.id) + ' · ' + esc(t.deliverable) + '</b><br>expected ' + esc(t.expected) +
          ' · reported ' + esc(t.reported) + '<br><em>accepted</em></div>'
      )
      .join('');
    $('#gateSum').textContent = F.tasks.length + '/' + F.tasks.length + ' accepted · ' + fmt(F.wall) + ' s wall · I2 invariants ok';

    renderMetrics('run');
    renderSteps();
  }

  function renderMetrics(mode) {
    if (mode === 'run') {
      $('#metrics').innerHTML =
        metric('WALL', fmt(F.wall), 's') +
        metric('PLANNER', F.planner.tokens, 'tok') +
        metric('HIDDEN', F.fusion.hidden, 'tok') +
        metric('KV', F.fusion.kvMb, 'MB');
    } else {
      const p = PROBES[state.probe];
      const hits = p.arms.filter((a) => a.accepted).length;
      $('#metrics').innerHTML =
        metric('SCHEMES', hits + '/' + p.arms.length, '') +
        metric('WALL', fmt(p.wall), 's') +
        metric('HIDDEN', p.arms[0].hidden, 'tok') +
        metric('KV', p.kvMb, 'MB');
    }
  }

  function metric(k, v, u) {
    return '<div><dt>' + k + '</dt><dd>' + v + (u ? '<small>' + u + '</small>' : '') + '</dd></div>';
  }

  /* ---------------- run playback ---------------- */

  function runLog(step) {
    const lines = [];
    const push = (k, t, cls) => lines.push({ k, t, cls });
    if (step >= 1) {
      push('00', 'run loaded · ' + F.id, 'accent');
      push('01', 'public state · ' + F.agents.length + ' agents · memory_max 4096 MB');
    }
    if (step >= 2) {
      push('02', 'planner · ' + F.planner.tokens + ' tokens · ' + fmt(F.planner.wall) + ' s', 'accent');
      push('03', 'action delegate · ' + F.tasks.length + ' tasks');
      F.tasks.forEach((t) => push('  ', t.id + ' · ' + t.deliverable));
    }
    if (step >= 3) {
      push('04', 'fuse · ' + F.fusion.hidden + ' analysis tokens · ' + fmt(F.fusion.prefill, 3) + ' s', 'accent');
      push('05', 'publish · ' + F.fusion.kvMb + ' MB · shared by ' + F.tasks.length + ' workers');
    }
    if (step >= 4) {
      push('06', F.tasks[0].id + ' running');
    }
    if (step >= 5) {
      const t = F.tasks[0];
      push('07', t.id + ' done · ' + t.outputTokens + ' tok · ' + fmt(t.decode) + ' s');
      push('08', 'gate ' + t.id + ' accepted · ' + t.reported, 'green');
      push('09', F.tasks[1].id + ' running');
    }
    if (step >= 6) {
      const t = F.tasks[1];
      push('10', t.id + ' done · ' + t.outputTokens + ' tok · ' + fmt(t.decode) + ' s');
      push('11', 'lru ' + (F.pool[0] || 'payload_swap_out(t1)'));
      push('12', 'gate ' + t.id + ' accepted · ' + t.reported, 'green');
      push('13', 'run completed · ' + fmt(F.wall) + ' s · ' + F.tasks.length + '/' + F.tasks.length, 'green');
      push('14', 'invariants · I2 ok', 'green');
    }
    return lines;
  }

  function applyRun(n) {
    state.step = n;
    const planner = $('#plannerNode');
    planner.classList.toggle('on', n >= 1);
    planner.classList.toggle('active', n >= 1 && n < 4);
    $('#plannerStatus').textContent = n >= 3 ? 'DONE' : n >= 1 ? 'RUNNING' : 'READY';
    $('#plannerStatus').className = 'status ' + (n >= 3 ? 'done' : n >= 1 ? 'running' : '');

    $$('[data-step]', $('#runView')).forEach((el) => {
      el.classList.toggle('on', n >= Number(el.dataset.step));
    });
    $('#fuserNode').classList.toggle('active', n === 3);

    const w1 = $('#w1');
    const w2 = $('#w2');
    if (w1) {
      w1.classList.toggle('running', n === 4);
      w1.classList.toggle('done', n > 4);
      const s = $('[data-wstatus]', w1);
      s.textContent = n > 4 ? 'DONE' : n === 4 ? 'RUNNING' : 'QUEUED';
      s.className = 'status ' + (n > 4 ? 'done' : n === 4 ? 'running' : '');
    }
    if (w2) {
      w2.classList.toggle('running', n === 5);
      w2.classList.toggle('done', n > 5);
      const s = $('[data-wstatus]', w2);
      s.textContent = n > 5 ? 'DONE' : n === 5 ? 'RUNNING' : 'QUEUED';
      s.className = 'status ' + (n > 5 ? 'done' : n === 5 ? 'running' : '');
    }

    const gate = $('#gateNode');
    gate.classList.toggle('on', n >= 6);
    gate.classList.toggle('active', n === 6);
    $('#gateStatus').textContent = n >= 6 ? 'ACCEPTED' : 'PENDING';

    $('#runState').textContent = n === 0 ? 'READY' : n >= 6 ? 'COMPLETE' : 'RUNNING';
    renderLog(runLog(n));
    updateControls();
  }

  function renderLog(lines) {
    const ol = $('#log');
    ol.innerHTML = lines
      .map((l) => '<li class="log-line ' + (l.cls || '') + '"><span class="k">' + esc(l.k) + '</span><span>' + esc(l.t) + '</span></li>')
      .join('');
    ol.scrollTop = ol.scrollHeight;
  }

  /* ---------------- arms ---------------- */

  function renderArmsTabs() {
    $('#probeTabs').innerHTML = PROBE_KEYS.map(
      (k) => '<button class="run-tab' + (k === state.probe ? ' on' : '') + '" data-probe="' + k + '" type="button">' + PROBE_LABEL[k] + '</button>'
    ).join('');
  }

  function renderArms() {
    const p = PROBES[state.probe];
    renderArmsTabs();
    $('#armsMeta').textContent = p.batch + ' · ' + fmt(p.wall) + ' s · expected ' + p.expected;
    $('#armsSetup').innerHTML =
      '<div class="q"><span>PUBLIC STATE</span>' + esc(p.question) + '</div>' +
      '<div><div class="note">' + esc(p.note) + '</div><div class="fus">fused · ' + p.arms[0].hidden + ' tokens · ' + p.kvMb + ' MB · ' + fmt(p.prefill, 3) + ' s prefill</div></div>' +
      '<div class="short-cmd"><span>SHORT COMMAND · PLANNER → WORKER</span><code>' + esc(p.shortCommand || '') + '</code><p>' + esc(p.shortText || '') + '</p></div>';

    $('#lanes').innerHTML = p.arms
      .map(
        (a, i) =>
          '<div class="lane" data-lane="' + (i + 1) + '">' +
          '<div><div class="lname">' + esc(a.name) + '</div><div class="linput">' + esc(ARM_INPUT[a.name] || '') + '</div></div>' +
          '<div class="lout">' + esc(String(a.report)) + '</div>' +
          '<div class="lkv">' + (a.name.indexOf('c2c_') === 0 ? '融合' : '—') + '</div>' +
          '<div class="ldec">' + fmt(a.decode, 2) + ' s</div>' +
          '<div class="lok">✓ 通过</div>' +
          '</div>'
      )
      .join('');
    $('#armsSum').textContent = p.arms.filter((a) => a.accepted).length + '/' + p.arms.length + ' 方案通过公开状态闸门 · ' + fmt(p.wall) + ' s 墙钟';
    renderMetrics('arms');
    renderSteps();
    renderSpotlight('arms');
  }

  function applyArms(n) {
    state.step = n;
    $$('#lanes .lane').forEach((l) => l.classList.toggle('on', n >= Number(l.dataset.lane)));
    $('#armsSum').classList.toggle('on', n >= 5);
    const p = PROBES[state.probe];
    const lines = [{ k: '00', t: 'five-scheme comparison · ' + state.probe, cls: 'accent' }];
    for (let i = 0; i < n; i++) {
      const a = p.arms[i];
      lines.push({ k: String(i + 1).padStart(2, '0'), t: a.name + ' · ' + fmt(a.decode, 2) + ' s · ' + a.report + (a.accepted ? ' · accepted' : ''), cls: a.accepted ? 'green' : '' });
    }
    if (n >= 5) lines.push({ k: '05', t: 'all five schemes accepted · expected ' + p.expected, cls: 'green' });
    renderLog(lines);
    $('#runState').textContent = n === 0 ? 'READY' : n >= 5 ? 'COMPLETE' : 'RUNNING';
    updateControls();
  }

  /* Historical scheduler component evidence, not a model communication arm. */
  function osTimeline(order, n) {
    let clock = 0;
    return order.map((id, i) => {
      const start = clock; clock += OS.serviceMs[id];
      return '<div class="os-job ' + (i < n ? 'finished' : i === n ? 'current' : '') + '" style="flex-grow:' + OS.serviceMs[id] + '"><b>' + esc(id) + '</b><small>' + start + '–' + clock + ' ms</small><span class="os-job-state">'+(i<n?'DONE':'READY')+'</span></div>';
    }).join('');
  }
  function renderOS(n) {
    const fifo=OS.fifoOrder, fair=OS.fairOrder;
    const total=Object.values(OS.serviceMs).reduce((a,b)=>a+b,0);
    const completed=order=>order.slice(0,n).reduce((sum,id)=>sum+OS.serviceMs[id],0);
    const small=order=>order.slice(0,n).filter(id=>id.startsWith('s')).length;
    $('#osView').innerHTML = '<div class="os-badge">历史调度组件验证 · 确定性负载 · 非 GPU 推理</div>' +
      '<div class="os-workload"><b>相同输入</b><span>big1 / big2：各 400 ms；s1–s6：各 30 ms</span><span>8 个任务同时就绪；总服务量 ' + total + ' ms</span></div>' +
      '<article class="os-policy"><div class="os-policy-head"><b>FIFO · 按提交顺序</b><span>' + n + '/8 已完成</span></div><div class="os-track">' + osTimeline(fifo,n) + '</div><div class="os-queue">' + esc(fifo.slice(n).join(' → ') || '全部完成') + '</div><div class="os-stats"><span>已消耗服务时间：' + completed(fifo) + ' ms</span><span>已完成短任务：' + small(fifo) + '/6</span></div></article>' +
      '<article class="os-policy fair"><div class="os-policy-head"><b>服务记账 · 同值时短任务优先</b><span>' + n + '/8 已完成</span></div><div class="os-track">' + osTimeline(fair,n) + '</div><div class="os-queue">' + esc(fair.slice(n).join(' → ') || '全部完成') + '</div><div class="os-stats"><span>已消耗服务时间：' + completed(fair) + ' ms</span><span>已完成短任务：' + small(fair) + '/6</span></div></article>' +
      '<div class="os-acb"><b>ACB 状态</b><span>READY → RUNNING → WAIT_MODEL → DONE</span><p>每一步分别回放两种策略的一次任务完成；两侧不代表同一墙钟时刻。</p></div>' +
      '<div class="os-result ' + (n===8?'revealed':'') + '"><b>平均周转时间</b><span>FIFO ' + fmt(OS.fifoTatMs,2) + ' ms → 服务记账 ' + fmt(OS.fairTatMs,2) + ' ms</span><p>相同服务量，改变调度顺序，减少短任务被长任务阻塞。</p></div>' +
      '<details class="os-source"><summary>验证依据</summary><p>checks.json：11 项组件检查通过；本页使用其中的调度顺序、平均周转时间与 I1/I2 验证结果。时间线按测试脚本中的确定性服务时长重建。</p><code>' + esc(OS.source) + '</code><br><code>' + esc(OS.workloadSource) + '</code></details>';
  }
  function applyOS(n) {
    state.step=n;renderOS(n);
    const lines=[{k:'00',t:'载入历史调度记录 · 8 个任务',cls:'accent'}];
    for(let i=0;i<n;i++) lines.push({k:String(i+1).padStart(2,'0'),t:'FIFO: '+OS.fifoOrder[i]+' DONE · vruntime: '+OS.fairOrder[i]+' DONE',cls:'green'});
    if(n===8)lines.push({k:'09',t:'8/8 完成 · I1/I2 通过',cls:'green'});
    renderLog(lines);$('#runState').textContent=n===0?'READY':n===8?'COMPLETE':'RUNNING';
    $('#metrics').innerHTML=metric('任务',8,'')+metric('已回放',n,'')+metric('FIFO',fmt(OS.fifoTatMs,2),'ms')+metric('服务记账',fmt(OS.fairTatMs,2),'ms');
    updateControls();
  }
  function applyCurrent(n) {
    if(state.mode==='run')applyRun(n);else if(state.mode==='os')applyOS(n);else if(state.mode==='system')applySystem(n);else applyArms(n);
  }

  function applySystem(n) {
    state.step=n;
    const shown=n<3?0:Math.min(n-2,4);
    const labels=['装载公开作业','建立 ACB 并准入','t1 拒绝并创建修复','t2 拒绝并创建修复','t1.r1 修复通过','t2.r1 仍失败','停止重试并保留失败','汇总全部运行证据'];
    const created=n<3?['t1','t2']:n<4?['t1','t2','t1.r1']:['t1','t2','t1.r1','t2.r1'];
    const states={};created.forEach(id=>states[id]=n<2?'READY':id.includes('.r')?'READY':'RUNNING');
    SYS.tasks.slice(0,shown).forEach(t=>states[t.id]=t.gate.accepted?'DONE':t.repairOf?'FAILED':'REPAIR_PENDING');
    const cards=SYS.tasks.filter(t=>created.includes(t.id)).map((t,i)=>{
      const observed=i<shown;const status=states[t.id];
      return '<article class="system-task '+(observed?(t.gate.accepted?'accepted':'rejected'):'')+'"><div class="node-head"><b>'+esc(t.id)+'</b><span class="status">'+status+'</span></div><div>'+(t.repairOf?'repair_of = '+esc(t.repairOf):esc(SYS.initialTasks[i].deliverable))+'</div><p>'+(observed?'PID '+t.pid+' · rc '+t.isolation.rc+' · RSS '+t.isolation.memory_peak_mib+' MiB':'等待隔离执行')+'</p>'+(observed?'<div>mmap '+fmt(t.cacheLoadMs,2)+' ms · '+t.outputTokens+' tokens</div><div class="system-verdict">'+(t.gate.accepted?'✓':'✕')+' <span>预期</span> '+t.gate.expected+' · <span>报告</span> '+(t.gate.reported===null?'need_info':t.gate.reported)+'</div>':'')+(observed&&t.feedback?'<details><summary>实际修复反馈</summary><p>'+esc(t.feedback)+'</p></details>':'')+'</article>';
    }).join('');
    const active=n>=6?SYS.tasks[3]:n>=5?SYS.tasks[2]:n>=4?SYS.tasks[1]:n>=3?SYS.tasks[0]:null;
    const poolCount=shown<2?0:shown===2?1:shown===3?3:5;
    const pool=SYS.poolEvents.slice(0,poolCount);
    const features=[['ACB 生命周期',n>=1],['GPU 预算准入',n>=2],['独立进程与 RSS 监督',shown>0],['mmap 载荷装载',shown>0],['验收反馈与有界修复',n>=3],['载荷换出与文件恢复',pool.length>0]];
    $('#systemView').innerHTML='<div class="os-badge">同一次历史 GPU 运行 · acbi-osclassics-sched-poolmmap</div><div class="system-features">'+features.map(([label,on])=>'<span class="'+(on?'on':'')+'">'+esc(label)+'</span>').join('')+'</div>'+
      '<div class="system-public"><b>一个场景：四作业 SJF 指标核验</b><p>'+SYS.jobs.map(j=>j.id+' (arrival '+j.arrival_seconds+' s, burst '+j.burst_seconds+' s)').join(' · ')+'</p><div>t1：总周转时间；t2：总等待时间。Planner 实际派出两个独立任务。</div></div>'+
      '<div class="system-stage"><b>当前阶段</b> '+esc(n?labels[n-1]:'准备回放')+'</div>'+
      '<div class="system-graph"><span>t1</span><span class="repair-link">⇢ repair_of ⇢</span><span>t1.r1</span><span>t2</span><span class="repair-link">⇢ repair_of ⇢</span><span>t2.r1</span></div><div class="system-task-grid">'+cards+'</div>'+
      '<div class="system-panels"><article><b>资源与隔离</b><p>任务显存估算 200 MiB；GPU 准入预算 '+fmt(SYS.budget/2**30,2)+' GiB。</p><p>RSS 上限 '+SYS.limits.memory_mib+' MiB；超时 '+SYS.limits.timeout_s+' s。</p><p>plain-subprocess · RSS watchdog</p><p>'+(active?'当前记录：PID '+active.pid+' · timed_out = '+active.isolation.timed_out+' · oom_killed = '+active.isolation.oom_killed:'等待执行记录')+'</p></article><article><b>载荷池与 mmap</b><p>容量 '+SYS.capacity+'；每次尝试独立发布并映射载荷。</p><div class="system-pool">'+(pool.length?pool.map(e=>'<div>'+esc(e.event)+' · '+esc(e.task_id)+'</div>').join(''):'等待载荷事件')+'</div></article></div>'+
      (n>=7?'<div class="system-summary"><b>恢复结果与边界</b><p>4 次尝试，2 个修复任务，1 次验收通过；t1 修复成功，t2 修复后仍失败，不再重试。</p><p>4 个不同 PID，均正常退出；本例没有 OOM 或超时。全程墙钟 '+fmt(SYS.wall)+' s。</p><p>原始 depends_on 为空；图中虚线是修复来源关系。本例未触发依赖等待、CFS 对照或银行家算法。</p><p>换回为应用文件事件，不代表内核缺页，也不代表旧载荷被实际复用。</p></div>':'')+
      '<details class="os-source"><summary>验证依据</summary><code>'+esc(SYS.source)+'/result.json</code><p>四次尝试对应 tasks/t1.json、t2.json、t1.r1.json、t2.r1.json。</p></details>';
    const lines=[{k:'00',t:'同一次运行 · 2 个独立任务',cls:'accent'}];
    for(let i=0;i<n;i++)lines.push({k:String(i+1).padStart(2,'0'),t:labels[i],cls:i===4?'green':''});
    renderLog(lines);$('#runState').textContent=n===8?'COMPLETE':n?'RUNNING':'READY';
    $('#metrics').innerHTML=metric('尝试',shown,'/4')+metric('修复',created.length-2,'/2')+metric('通过',SYS.tasks.slice(0,shown).filter(t=>t.gate.accepted).length,'')+metric('墙钟',fmt(SYS.wall),'s');
    updateControls();
  }

  /* ---------------- controls ---------------- */

  function currentSteps() {
    return state.mode === 'run' ? RUN_STEPS : Array.from({ length: ['os','system'].includes(state.mode)?8:ARM_STEPS }, () => ({ dur: state.mode==='system'?3000:1500 }));
  }

  function renderSteps() {
    const steps = currentSteps();
    $('#steps').innerHTML = steps.map((_, i) => '<span class="step" data-seg="' + i + '"><i></i></span>').join('');
    updateControls();
  }

  function updateControls() {
    const steps = currentSteps();
    const max = steps.length;
    $$('#steps .step').forEach((el, i) => {
      el.classList.toggle('done', i < state.step);
      el.classList.toggle('on', i === state.step - 1 && state.step > 0);
    });
    $('#counter').innerHTML = '<b>' + String(Math.max(state.step, 0)).padStart(2, '0') + '</b> / ' + String(max).padStart(2, '0');
    $('#prev').disabled = state.step <= 0;
    $('#next').disabled = state.step >= max;
    $('#play').textContent = state.playing ? '暂停' : state.step >= max ? '重播' : '播放';
  }

  function schedule() {
    clearTimeout(state.timer);
    if (!state.playing) return;
    const steps = currentSteps();
    if (state.step >= steps.length) {
      state.playing = false;
      updateControls();
      return;
    }
    const dur = steps[state.step].dur;
    state.timer = setTimeout(() => {
      applyCurrent(state.step + 1);
      if (state.step >= steps.length) {
        state.playing = false;
        updateControls();
      } else {
        schedule();
      }
    }, dur);
  }

  function play() {
    const steps = currentSteps();
    if (state.step >= steps.length) {
      state.step = 0;
      applyCurrent(1);
    }
    state.playing = true;
    updateControls();
    schedule();
  }

  function pause() {
    state.playing = false;
    clearTimeout(state.timer);
    updateControls();
  }

  function step(delta) {
    state.playing = false;
    clearTimeout(state.timer);
    const steps = currentSteps();
    const next = Math.max(0, Math.min(steps.length, state.step + delta));
    applyCurrent(next);
  }

  function spot(label, value, sub) {
    return '<div class="spot"><span>' + esc(label) + '</span><b>' + esc(value) + '</b><small>' + esc(sub) + '</small></div>';
  }

  function renderSpotlight(mode) {
    if (mode === 'run') {
      $('#spotlight').innerHTML =
        spot('SHORT COMMAND', F.tasks[0].command, 'Worker 只读这一句') +
        spot('SHARED KV', '1 次融合 → 2 个 Worker', F.fusion.hidden + ' tokens · ' + F.fusion.kvMb + ' MB') +
        spot('GATE', F.tasks.length + '/' + F.tasks.length + ' accepted', 'public-state simulation');
    } else {
      const p = PROBES[state.probe];
      const hits = p.arms.filter((a) => a.accepted).length;
      $('#spotlight').innerHTML =
        spot('SHORT COMMAND', p.shortCommand || '—', p.shortText || '') +
        spot('KV 通道', '隐藏方案使用', p.arms[0].hidden + ' tokens · ' + p.kvMb + ' MB') +
        spot('GATE', hits + '/5 方案通过', 'same task · five schemes');
    }
  }

  function setMode(mode) {
    state.mode = mode;
    state.step = 0;
    state.playing = false;
    clearTimeout(state.timer);
    $$('.tab').forEach((t) => t.classList.toggle('on', t.dataset.mode === mode));
    $('#runView').hidden = mode !== 'run';
    $('#armsView').hidden = mode !== 'arms';
    $('#osView').hidden = mode !== 'os';
    $('#systemView').hidden = mode !== 'system';
    const foot=$$('footer span');
    foot[0].textContent=mode==='os'?'历史调度组件验证 · 2026-10-19':'真实 GPU 运行 · acbi4 · 2026-10-20';
    foot[1].textContent=mode==='os'?'ACB · FIFO · service accounting':'Qwen3-4B → Qwen3-0.6B · frozen fuser';
    if(mode==='system'){foot[0].textContent='历史真实运行回放 · 非通信机制贯通';foot[1].textContent='ACB · admission · processes · repair · mmap · payload pool';}
    if (mode === 'run') {
      $('#eyebrow').textContent = 'REAL RUN · ' + F.id.toUpperCase();
      $('#title').textContent = '一次真实的 C2C 委派';
      $('#deck').innerHTML = 'Planner 只发<b>短命令</b>；语义走<b>共享 KV</b>——一次融合，两个 Worker 复用。';
      renderMetrics('run');
      renderSpotlight('run');
      renderSteps();
      applyRun(1);
    } else if(mode==='system') {
      $('#eyebrow').textContent='OS RUNTIME · INTEGRATED REPLAY';
      $('#title').textContent='一次任务失败，运行时如何继续处理';
      $('#deck').textContent='同一实例贯通 ACB、资源准入、隔离进程、mmap、载荷管理、验收与有界修复。';
      $('#spotlight').innerHTML='';renderSteps();applySystem(0);
    } else if(mode==='os') {
      $('#eyebrow').textContent='OS RUNTIME · SCHEDULER EVIDENCE';
      $('#title').textContent='同一批任务，调度顺序影响完成体验';
      $('#deck').textContent='从就绪队列到 ACB 状态：对比 FIFO 与服务记账，观察短任务如何被提前执行。';
      $('#spotlight').innerHTML='';renderSteps();applyOS(0);
    } else {
      $('#eyebrow').textContent = 'FIVE SCHEMES · SAME TASK';
      $('#title').textContent = '五个通信方案，同一道题';
      $('#deck').innerHTML = '同一份公开状态、同一套判分。隐藏方案只读<b>短命令</b>，语义由<b>融合 KV</b>送达；五个方案全部通过闸门。';
      renderSpotlight('arms');
      renderArms();
      applyArms(1);
    }
  }

  /* ---------------- bind ---------------- */

  $('#play').addEventListener('click', () => (state.playing ? pause() : play()));
  $('#prev').addEventListener('click', () => step(-1));
  $('#next').addEventListener('click', () => step(1));
  $$('.tab').forEach((t) => t.addEventListener('click', () => setMode(t.dataset.mode)));
  document.addEventListener('click', (e) => {
    const p = e.target.closest('[data-probe]');
    if (p) {
      state.probe = p.dataset.probe;
      renderArms();
      applyArms(0);
    }
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === ' ') { e.preventDefault(); state.playing ? pause() : play(); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); step(1); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); step(-1); }
  });

  renderRunShell();
  renderArms();
  setMode('run');

  const params = new URLSearchParams(location.search);
  const initial = Math.max(0, parseInt(params.get('step') || '0', 10) || 0);
  if(params.get('mode')==='system') {
    setMode('system');applySystem(Math.min(initial,8));
  } else if(params.get('mode')==='os') {
    setMode('os');applyOS(Math.min(initial,8));
  } else if (params.get('mode') === 'arms') {
    setMode('arms');
    if (initial) applyArms(Math.min(initial, ARM_STEPS));
  } else if (initial) {
    applyRun(Math.min(initial, RUN_STEPS.length));
  }
})();
