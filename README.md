# C2CAIOS

**用操作系统设计思想构建多智能体 LLM 系统：智能体之间以共享 KV cache 状态（C2C）而非文字交接来传递昂贵语义，并由操作系统风格的调度、隔离与资源管理机制支撑运行。**

## 设计思想

系统由一个冻结的 **Qwen3-4B Planner** 与一个 **Qwen3-0.6B Worker** 池组成。Planner 观察公开运行时状态，自主决定是否把有界核验任务委派给 Worker。两个 Agent 之间的信息有两条通道：

- **文字通道**：Planner 的分析笔记（或重新生成的交接摘要）被序列化成文字，由接收方重新预填，按量付费。
- **C2C 通道**：Planner 分析笔记的**真实自回归 KV cache** 经已发布的 fuser 转换后直接融合进 Worker 上下文。Worker 只看得到一条短命令，昂贵语义以 KV 状态到达。

围绕这两条通道实现八个对照臂（text_full / text_brief / state_native / c2c_shared / **c2c_hidden_analysis** / c2c_hidden_full / c2c_state_hidden / c2c_command），在同一 Planner 产物下做苹果对苹果的比较。

## 操作系统机制映射

| OS 概念 | 本系统机制 |
|---|---|
| 进程控制块（PCB） | `AgentControlBlock` 状态机：READY / RUNNING / WAIT_MODEL / WAIT_FUSION / BLOCKED_ON_DEP / DONE / FAILED，等待时长按态记账，`repair_of` 溯源 |
| CFS 公平调度 | `FairScheduler`（min-vruntime + 服务时间记账）；实测 FIFO 829 ms → CFS 274 ms 平均周转 |
| 死锁检测与避免 | 等待图环检测（EDEADLK 快速失败）+ 银行家安全序准入 |
| 进程隔离 | `os_isolate`：真实 OS 子进程 + 配额限额，单 Agent 故障被包含为 FAILED，不级联 |
| cgroup / OOM killer | 模拟 cgroup v2 文件系统（memory.max/current/peak、pids.max、memory.events），真实遥测驱动处决，有界修复闭环 |
| 消息传递 vs 共享内存 | 文字交接（序列化-传输-反序列化）↔ C2C KV 融合（一次融合、N 个 Worker 复用） |
| 系统调用 / 能力闸门 | 语义闸门对照公开状态验收 Worker 产物：采纳 / 拒绝 / 有界修复，Worker 不能自宣布完成 |
| 上下文管理 | LRU payload 池（swap-out 文件事件）+ `--mmap` 主机零拷贝（0.092s→0.025s，约 3.7×） |

## 关键实测结果（RTX 4080 SUPER，模型冻结，离线评分，参考答案不上计算端）

- **质量平价**：30 个 OS 主题场景上，Worker 只读短命令的隐藏 C2C 臂取得 **23/30**，与最优臂并列且不低于读全文的 text_full（21/30）；在 Planner 写出答案的 22 题子集上，隐藏臂 **22/22 ≥ 文字臂 20/22**。
- **交接成本**：一次 C2C 融合约 0.1 s、被全部 Worker 共享；text_brief 通道每次交接要真实重新生成 handoff 文字（约 59 s）——C2C 交接成本低约 **600×**（口径说明：KV 字节数 ≠ 网络提速）。
- **多 Worker 摊销**：同一已融合 payload 被多个 Worker 真实消费（2.2–2.4×）；30 题多智能体批次中 Planner 自然产生 18 单 / 9 双 / 3 三的实际委派分布。
- **背景可扩展性**：文字通道预填成本随背景线性增长（21.7k token 背景时 32G 显存 OOM），C2C 通道与背景规模无关、保持恒定。

**诚实边界**：GPU 设备级零拷贝、systemd 全权环境下的内核级 cgroup 实证未实现；已实现的均为真实测量，未测不宣称。

## 目录结构

```
src/agent_runtime/    运行时本体（唯一活跃包；统一入口 scripts/run.py）
scripts/              运行 / 离线评分 / 场景生成脚本
configs/              模型 pin（cloud.models.lock.json）与协议配置
data/c2c-core-os/     公开场景输入（不含任何参考答案）
docs/                 里程碑证据、方法论映射、各批次结果报告
demo/                 委派过程演示站与仪表盘数据
third_party/          钉版第三方代码（C2C fuser 接口、监督源）
```

## 复现

模型版本固定在 `configs/cloud.models.lock.json`（Qwen/Qwen3-4B、Qwen/Qwen3-0.6B、nics-efc/C2C_Fuser，均为公开仓库）。示例运行：

```bash
python scripts/run.py --c2c-core-rich-hidden --arith-scaffold \
  --public-filename data/c2c-core-os/public-probe3c-admission-mem-01.json \
  --output-dir outputs/example
```
