# 07 Agent 协同协议

Agent 不是一个"调用一次就出结果"的接口，而是一个需要工程化包装的不确定过程。

本文定义谁做什么、产出什么格式、怎么验收、失败怎么办、什么时候放弃。没有这套协议，"Agent 协同设计"就只是一句话。

---

## 1 需要协议的原因

三个问题必须解决。

Agent 的输出是自由文本，而下游需要的是 `ku.json`、`.so` 和 `params.bin`。中间的转换必须由代码完成，不能靠再问一次 Agent。

Agent 会出错，也会针对验收标准取巧。验收必须由确定性检查完成，而且检查维度要对 Agent 隐藏，理由见 09 文档第 5.3 节。

Agent 的失败原因很分散。没有结构化的失败分类和对应的下一步动作，重试只是浪费预算。

协议的目标是把 Agent 的不确定性放进一个有明确边界、明确验收和明确预算的流程里。

---

## 2 角色划分

| 角色 | 是否 LLM | 职责 | 产出 | 权限边界 |
|---|---|---|---|---|
| 协调者 | 否，确定性代码 | 驱动状态机、管理预算、调用检查、记录日志 | 状态转移 | 不生成内容 |
| 分组 Agent | 是 | 决定融合分组 | `FusionDecision` JSON | 不写代码 |
| 内核 Agent | 是 | 实现单个 KernelUnit | `src/impl.*` 加 `ku.json` | 只在脚手架内改文件 |
| 调优 Agent | 是 | 搜索参数、选择实现 | `TuningReport` | 只能改 params 与 impl 选择 |
| 审计 Agent | 是 | 独立找问题，产出风险清单 | `AuditReport` | 只读，不改产物 |
| 人类 | 否 | 门禁审批、疑难介入 | 批准或拒绝 | 可以改一切 |

只有一个角色是确定性的，但它掌握所有状态转移的权限。这保证 Agent 无论多强都无法绕过检查、无法改变流程。

审计 Agent 单独跑一轮，是因为同一个 Agent 写完代码后立刻自查，倾向于确认自己是对的。独立的一轮、独立的上下文、明确要求"你的任务是找问题"，能提高召回率。

但审计 Agent 不构成最终门禁。最终门禁是确定性检查和人类评审，见第 5 节与第 8 节。

---

## 3 产物约定

第一条规则：Agent 的任何产出都必须是文件，且必须通过 schema 校验。不接受"在对话里说完成了"。

| 产物 | 格式 | schema 校验 | 确定性检查 |
|---|---|---|---|
| `FusionDecision` | JSON | 是 | D1 到 D6，见 08 文档第 5.2 节 |
| `ku.json` | JSON | 是 | io 与 plan 一致、参数 schema 一致 |
| `src/impl.*` | 源码 | 无 | 编译加符号检查 |
| `TuningReport` | JSON | 是 | 参数在声明域内 |
| `AuditReport` | JSON | 是 | 引用的文件存在 |
| `ConformanceReport` | JSON，引擎产出 | 是 | 无 |

每个产物都要携带元信息：

```json
{
  "schema": "loom/fusion-decision/1",
  "produced_by": {"agent": "claude-...", "prompt_id": "partition-v3", "seed": 42},
  "inputs_hash": "sha256:...",
  "created": "2026-10-20T10:00:00Z"
}
```

`inputs_hash` 用于防止过期的产物被误用。模型或 golden 变化后，旧产物的 hash 不匹配，自动失效。

---

## 4 闭环状态机

状态按顺序推进：INGESTED、PRECISION_PLANNED、PARTITIONED、EMITTED、GENERATING、CHECKING、TUNING、AUDITED、PACKED，最后登记到 registry。

有两处回退。分组结果没有通过 D1 到 D6 校验时，退回分组 Agent 重新给出建议；CHECKING 未通过时，退回 GENERATING 带着反馈重写该组实现。

各步骤的准入条件：

| 转移 | 准入条件 | 失败时的动作 |
|---|---|---|
| 到 PRECISION_PLANNED | 已记录数值方案、模型基线与质量预算 | 修正离线策略或参考覆盖，不允许 runtime 隐式猜测 |
| 到 PARTITIONED | D1 到 D6 全过，分组数不超预算 | 把失败的规则编号和违规节点交给分组 Agent 重试一次，再失败则回退到规则策略 |
| 到 EMITTED | plan 通过引擎静态校验 | 报 Kernel Harness 的 bug，这不是 Agent 的问题 |
| 到 CHECKING | 编译成功、符号正确、L1 通过 | 按第 6 节的分类处理 |
| 到 TUNING | L1 与 L2 都通过 | 视为 CHECKING 失败 |
| 到 AUDITED | 审计无高风险项，确定性检查通过 | 人工介入 |
| 到 PACKED | 签名与许可扫描通过 | 修复 |

每次状态转移都要落盘一个产物和一条日志，任何一次失败都能精确复现。

内层是单组循环，也是 Agent 花时间最多的地方：

1. 生成或修改 `src/impl.cu`。
2. 执行 `check.sh --level L1`。编译失败则喂回截断后的错误，运行崩溃则喂回 canary 报告，数值不对则喂回 `ConformanceReport`，通过则进入下一步。
3. 执行 `check.sh --level L2`，做形状扫描。
4. 执行 `bench.sh`，与 baseline 比较并给出瓶颈分类。偏慢则喂回诊断，达标则退出循环。
5. 尝试次数加一，超过预算则升级。

---

## 5 验收

Agent 说"完成了"没有意义。按权重排序的判据如下：

| 权重 | 判据 | 由谁执行 |
|---|---|---|
| 40% | L1 差分校验，目标数值契约的独立参考与比较规则 | 引擎，确定性 |
| 20% | L2 形状与输入分布扫描 | 引擎，确定性 |
| 15% | 结构检查，含 canary、对齐扰动、依赖探针、确定性 | 引擎，确定性 |
| 15% | 性能达标 | 引擎加脚本 |
| 10% | 人类评审，看可读性、可维护性、许可 | 人 |

85% 是机器判据。这是有意的：Agent 的产出必须能被机器以压倒性的证据判定好坏，否则规模化不成立。

---

## 6 失败分类与修复建议

Agent 常见的问题是反复犯同一类错误。对应的下一步动作是固定的，写死在协调者里，不交给 LLM 决定，因为这部分属于工程知识。

| 失败 | 检测方式 | 反馈给 Agent 的内容 | 典型原因 |
|---|---|---|---|
| `COMPILE_ERROR` | 编译 | 截断后的错误，以及最近三个相似 kernel 的写法 | API 用错、类型不匹配 |
| `SYMBOL_MISSING` | `dlsym` 失败 | 入口必须是 `extern "C"` 且不被 strip，参考模板 | 忘记 `extern "C"` |
| `OOB` | canary 或 sanitizer | 越界的 slot 与索引位置 | 缺少边界保护 |
| `WRONG_VALUE` | L1 差分 | 完整报告，含错误的空间分布 | 边界处理、stride 理解错误、漏用输入 |
| `IGNORED_INPUT` | 依赖探针 | 哪个输入被扰动后输出没有变化 | 漏了一个分支 |
| `SLOW` | bench 加 roofline | 瓶颈分类与相应建议 | 分块不当、未向量化 |
| `NONDET` | 两次比对 | 两次输出不同，检查 atomic 归约顺序 | 归约顺序不固定 |
| `TIMEOUT` | 看门狗 | 执行超过时限，检查循环终止条件 | 死循环 |
| `ILLEGAL_ALIGN` | 对齐扰动 | 基址偏移 4 字节时崩溃，不要假设 256 字节对齐 | 强制类型转换前未检查对齐 |
| `SMEM_OVERFLOW` | 编译期或运行期 | 需要多少动态 smem，超过声明的上限 | tile 太大 |
| `REGISTER_SPILL` | ptxas 报告 | 寄存器用满并发生溢出，减小 tile 或拆分 | 寄存器压力过大 |
| `TOLERANCE_GAMING` | 误差分布分析 | 误差集中在最大元素上，疑似低精度累加 | 用 fp16 累加 |
| `SCOPE_VIOLATION` | 文件 diff | 修改了不允许修改的文件，已回滚 | Agent 改了 ref 或 manifest |

每条反馈都应当是具体的动作，而不是"再试一次"。

---

## 7 预算、重试与升级

### 7.1 预算

```yaml
per_kernel_unit:
  max_attempts: 8
  max_wall_clock: 45m
  max_gpu_minutes: 30
  max_tokens: 400k
per_model:
  max_wall_clock: 12h
  max_gpu_hours: 8
  max_total_cost_usd: 200
```

预算必须硬性截断，不是建议。超出后停止，产出当前最好的版本加完整诊断，交给人。

### 7.2 升级阶梯

```
第 1 到 3 次   同模型重试，带修复建议
第 4 到 6 次   换更强的模型，或更换表达方式（Triton 换 CUDA，或反向）
第 7 次       加入检索到的相似已成功 kernel 作为上下文
第 8 次       人工介入，附 debug bundle
```

debug bundle 的内容：

```
bundle.tar.gz
├── task.md                任务描述
├── attempts/              每次尝试的源码与报告
├── best/                  最好一次的完整产物
├── conformance.json       完整校验报告
├── profile.json           逐 node 耗时与瓶颈诊断
└── replay.bin             录制文件，可在本地复现
```

### 7.3 放弃条件

有明确的放弃条件比一直尝试更可取：

- 连续八次 L1 失败，且失败位置每次不同。这说明是能力问题而不是笔误。
- 性能在第四次尝试之后不再提升，且已经接近 roofline。
- 失败的形状只在 coverage 的边缘，可以把该形状从 coverage 中划掉，走 ref 降级。

最后一条值得强调：不是每个 kernel 都必须满足全域。允许"部分覆盖加降级"是合理的产品决策，前提是它被声明出来，即 `package.coverage` 加对应 `policy`。

---

## 8 人类在环

| 门禁 | 人类看什么 | 为什么不能全自动 |
|---|---|---|
| G1 分组方案 | 是否合理，是否漏掉明显机会 | 需要架构判断 |
| G2 首个 KernelUnit 通过 | 代码质量与可维护性 | 会影响后续所有 KernelUnit 的风格 |
| G3 性能达标 | 差距归因，是否继续投入 | 需要判断 roofline 是否可信 |
| G4 打包发布 | 许可、签名、coverage 声明是否如实 | 法律与信誉风险 |
| G5 接口变更 | 任何 S0 或 S1 的变更 | 影响所有历史包 |

G5 最重要：Agent 不允许修改引擎。如果 Agent 认为接口不好用，只能提出建议，由人决定是否变更 ABI。

---

## 9 可复现性

Agent 的不确定性是长期维护的主要障碍。应对方式：

| 手段 | 作用 |
|---|---|
| 固定 seed，尽量降低采样随机性 | 相同输入尽量得到相同输出 |
| 产物按内容寻址且不可变 | 一旦生成就冻结，"昨天好好的"可以精确重放 |
| provenance 记录模型版本、prompt hash、工具版本 | 可以追溯 |
| 生成的源码入库，不只存 `.so` | 可以重编译、可以审计、可以被下一代 Agent 改进 |
| 每次状态转移落盘产物 | 可以从中断处继续 |
| CI 保留最近三十次产物 | 回归问题可以二分 |

最重要的一条：不要把 Agent 放在生产加载路径上。生产的包是冻结产物，不是每次启动时生成的。这是对"运行期生成太贵"这一质疑的直接回答。

---

## 10 上下文工程

原则是给 Agent 的上下文不超过 20k token。超出说明问题没有切分好。

### 必须提供

```yaml
任务:
  - KernelUnit 名称与数学定义（ref.cpp，可读）
  - 输入输出声明，含 dtype、shape、layout 与符号取值范围
  - 验收标准：L1、L2、性能阈值、workspace 上限
  - 禁止修改的文件列表
环境:
  - GPU 型号、arch family、smem 与寄存器上限、SM 数量
  - 可用 DSL 版本
示例:
  - 一两个已通过的相似 kernel 完整源码
反馈:
  - 上一次尝试的完整报告，含定位、错误分布、疑似原因
```

### 不应提供

| 不提供 | 原因 |
|---|---|
| 完整模型图 | 会诱导越界思考，也浪费上下文 |
| 发布留出样本与独立门禁细节 | 开发反馈可包含容差；隐藏留出样本，Agent 不得修改比较规则 |
| 检查脚本源码 | 会诱导针对检查取巧 |
| 其他组未通过的尝试 | 噪声 |
| "尽量快"这类模糊目标 | 必须给具体数值阈值 |

不应提供完整模型图这一点值得展开：Agent 只看得到本组的约定，看不到整个模型。这是本架构的一个结构性优势，因为 plan 是显式的，每个 kernel 的任务是自包含的。在编译器方案里，没法让 LLM 优化一个融合区域而不给它整张图。

---

## 11 一个完整回合

```
[协调者] 状态 EMITTED。开始 group g17（C2f，node 12 到 15）。
[协调者] 生成脚手架 work/groups/g17/，ku.json 已从 manifest 填充。
[协调者] 交给内核 Agent，第 1 次尝试。

[kernel-agent]  写入 src/impl.cu，初版使用向量化读，未处理尾块。
[协调者] check.sh --group g17 --level L1
         编译失败：identifier __half2float2 undefined
[协调者] 反馈 COMPILE_ERROR，附错误与相似 kernel 片段。

[kernel-agent]  修正 API 调用。
[协调者] 编译通过，L1 失败。
         WRONG_VALUE @ node 17, slot 24, worst_index [0,71,118,42]
         错误集中在最后两行与最后两列，占 71%
         疑似边界处理问题：输入未 padding
[协调者] 反馈 WRONG_VALUE。

[kernel-agent]  增加边界保护。
[协调者] L1 通过，max_err 3.1e-4，小于容差 1.2e-3。
         L2 在 H=642 失败，该形状不在 coverage 内，记录后跳过。
         bench 1.42 ms，baseline 1.10 ms。
         roofline：memory-bound，带宽利用率 78%。
[协调者] 反馈 SLOW，memory-bound。

[kernel-agent]  减少中间搬运，直接消费 split 结果。
[协调者] L1 通过，bench 1.08 ms。
         确定性检查：两次运行逐位一致。
         对齐扰动：delta=4 时崩溃。
[协调者] 反馈 ILLEGAL_ALIGN。

[kernel-agent]  去掉强制 float4 转换，改为显式判断对齐。
[协调者] 全部通过，第 5 次尝试，进入 TUNING。
```

过程中每次失败都有具体位置和具体建议，Agent 修的是单点而不是重来一遍。

---

## 12 常见问题

| 做法 | 后果 |
|---|---|
| 让 Agent 直接改引擎代码 | 接口约定被侵蚀，ABI 腐化 |
| 让 Agent 生成 plan 连线 | 静默错误，调试成本极高 |
| 让 Agent 修改判据或用开发样本替代发布门禁 | 无法独立评估正确性与泛化 |
| 让 Agent 看到检查脚本 | 针对检查取巧 |
| 用再问一次 Agent 代替 schema 校验 | 不确定性叠加 |
| 失败后从头重来，不保留产物 | 浪费预算，且无法分析 |
| 无限重试 | 成本失控 |
| 只靠 Agent 自查 | 它总会说已完成 |
| 生产环境现场生成 | 延迟、成本、可靠性都会出问题 |
| 不给具体性能阈值，只说"优化" | Agent 无法判断何时停止 |
