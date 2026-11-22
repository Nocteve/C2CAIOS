// 真实运行留痕：artifacts/acbi-osclassics-fault-01（arm = c2c_hidden_analysis, seed 42）
// 场景：缺页与内存限额记账 · Qwen3-4B Planner → Qwen3-0.6B Worker

export interface AgentState {
  id: string;
  exitCode: number;
  peakRssMb: number;
  note: string;
}

export interface TaskTrace {
  id: string;
  label: string;
  command: string;
  deliverable: string;
  successCriteria: string;
  grantedFiles: string[];
  think: string;
  hitField: string;
  hitValue: string;
  decodeS: string;
  waitModelS: string;
  waitFusionS: string;
}

export const SCENARIO = {
  name: 'osclassics-fault-01 · 缺页与内存限额记账',
  arm: 'c2c_hidden_analysis',
  seed: 42,
  memoryLimitMb: 4096,
  wallS: '64.2 s',
  planner: {
    note:
      'The planner needs to determine which agent was killed by the memory limit... ' +
      'The memory limit is 4096 MB, and the peak RSS of the killed agent (A2) is 8200 MB, ' +
      'which exceeds the limit. The other agents (A1, A3, A4) completed normally with exit code 0. ' +
      'Therefore, the oom_agent_index is 1 (A2) and survived_siblings is 3.',
    tokens: 363,
    chars: 428,
    kvMb: 260,
    wallS: '26.1 s',
    brief:
      'compute oom_agent_index as the index of the agent with exit code 137 and peak_rss_mb > 4096, ' +
      'and compute survived_siblings as the count of agents with exit code 0',
  },
  hiddenTokens: 177,
  prefillS: '0.001 s',
  agents: [
    { id: 'A1', exitCode: 0, peakRssMb: 1024, note: '正常完成' },
    { id: 'A2', exitCode: 137, peakRssMb: 8200, note: '超过限额被杀（SIGKILL）' },
    { id: 'A3', exitCode: 0, peakRssMb: 2340, note: '正常完成' },
    { id: 'A4', exitCode: 0, peakRssMb: 1788, note: '正常完成' },
  ] as AgentState[],
  tasks: [
    {
      id: 't1',
      label: 't1 · 计算被杀代理序号',
      command:
        'compute oom_agent_index as the index of the agent with exit code 137 and peak_rss_mb > 4096',
      deliverable: 'oom_agent_index',
      successCriteria: 'oom_agent_index == 1',
      grantedFiles: ['runtime-state.json'],
      think: 'A2 退出码 137、峰值 RSS 8200 MB，超过 4096 MB 限额 → 被杀代理是 A2，序号为 1。',
      hitField: 'oom_agent_index',
      hitValue: '1',
      decodeS: '7.4 s',
      waitModelS: '33.76 s',
      waitFusionS: '0.799 s',
    },
    {
      id: 't2',
      label: 't2 · 计算存活兄弟数',
      command: 'compute survived_siblings as the count of agents with exit code 0',
      deliverable: 'survived_siblings',
      successCriteria: 'survived_siblings == 3',
      grantedFiles: ['runtime-state.json'],
      think: 'A1、A3、A4 均以退出码 0 正常完成 → 存活兄弟数为 3。',
      hitField: 'survived_siblings',
      hitValue: '3',
      decodeS: '5.6 s',
      waitModelS: '26.75 s',
      waitFusionS: '0.728 s',
    },
  ] as TaskTrace[],
};

export interface StepDef {
  id: string;
  no: string;
  title: string;
  durMs: number;
}

export const STEPS: StepDef[] = [
  { id: 'scene', no: '0', title: '场景装载：内存限额记账', durMs: 5000 },
  { id: 'plan', no: '1', title: 'Planner 规划（笔记只以 KV 存在）', durMs: 9000 },
  { id: 'brief', no: '2', title: '析出委派 brief（短命令）', durMs: 6000 },
  { id: 'c2c', no: '3', title: 'C2C 交接：KV 选区融合', durMs: 8000 },
  { id: 't1', no: '4', title: 'Worker 执行 t1', durMs: 7000 },
  { id: 't2', no: '5', title: 'Worker 执行 t2', durMs: 7000 },
  { id: 'gate', no: '6', title: '闸门校验与汇总', durMs: 6000 },
];
