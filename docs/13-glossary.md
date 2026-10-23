# 13 术语表

术语混乱会以"同一件事三种叫法"的形式渗透进代码与文档，最终变得不可维护。本表是唯一的权威定义。

---

## 1 核心概念

| 术语 | 英文 | 定义 | 出处 |
|---|---|---|---|
| 代码生成 | code generation | 从语义图产出 kernel 实现。本项目不做，交给 Agent 与 Kernel Harness | 00 文档 0.2 节 |
| 运行时 | runtime | 加载、内存分配、并发、执行、动态形状处理 | 同上 |
| 正确性保证 | correctness assurance | 校验、降级、复现、错误定位 | 同上 |
| 抽象层次 | abstraction level | 接口所处层次决定其寿命，层次越高，硬件特性变化越慢 | 00 文档 0.4 节 |
| 数据流与语义的边界 | — | 引擎理解数据流，不理解语义。全项目唯一不可协商的约束 | 00 文档 0.5 节 |
| 语义无关性测试 | semantics-agnostic test | 把 plan 中所有 ku 名称与 dtype 改成无意义名称后，引擎行为必须不变 | 00 文档 0.5 节 |
| 接口约定不依赖生产者 | — | 引擎对"谁生产了 kernel"保持无知，因此不随生产者变化 | 00 文档 0.7 节 |

---

## 2 运行时与执行

| 术语 | 英文 | 定义 | 出处 |
|---|---|---|---|
| plan | plan | 显式编排：slot、node、依赖、内存计划、variant | 05 文档 |
| slot | slot | 缓冲区槽位。一个张量值对应恰好一个 slot，视图也是 slot | 05 文档第 3 节 |
| node | node | 一次 kernel 调用，含 ku、impl、输入输出、参数、stream | 05 文档第 4 节 |
| variant | variant | 一组谓词、实现覆盖与内存计划，用于形状特化 | 05 文档第 5 节 |
| impl | implementation | KernelUnit 内部的一个具体实现，如 ref、fast、tuned | 04 文档第 2 节 |
| package | package | 引擎的输入单元：plan、kernels、weights、goldens、签名 | 06 文档 |
| fingerprint | fingerprint | 特化声明：这个包在什么硬件、驱动和 ABI 上验证过 | 06 文档第 4 节 |
| coverage | coverage | 包的形状覆盖声明，域外可执行但标记为未验证 | 06 文档第 3 节 |
| prepack | prepack plan | 加载时执行一次的权重转换 plan | 06 文档第 6 节 |
| in-flight 槽位 | in-flight slot | 一个 session 内可并发的 run 单元，各自持有独立 arena | 02 文档第 4.2 节 |
| 设备上下文失效 | device poisoning | 一次非法访存后整个 context 不可用，必须快速失败 | 09 文档第 7.2 节 |

---

## 3 KernelUnit

| 术语 | 英文 | 定义 | 出处 |
|---|---|---|---|
| KernelUnit | kernel unit | ABI 的最小可替换单元，一个 `.so` 与一个入口符号 | 04 文档第 2 节 |
| 入口符号 | entry symbol | `lm_ku_entry_v1` 与 `lm_ku_caps_v1` | 04 文档第 2 节 |
| caps | capabilities | 能力声明：backend、arch、dtype、对齐、确定性、容差 | 04 文档第 5 节 |
| 派生量 | derived | plan 声明、引擎求值、传给 kernel 的标量 | 04 文档第 4.3 节 |
| workspace | workspace | kernel 的临时显存，从共享 scratch arena 分配 | 04 文档第 9 节 |
| shim | shim | 把 DSL 产物（Triton、TileLang）包装成 `lm_ku_entry_v1` 的薄包装，由模板生成 | 08 文档第 8.3 节 |
| 厂商 KernelUnit | vendor unit | 内部调用 cuDNN 或 cuBLAS 的 KernelUnit，与自写实现地位相同 | 06 文档第 7 节 |
| 方言 | dialect | 源码所用 DSL：cuda_cpp、triton、tilelang、cute、ptx、prebuilt | 04 文档第 5 节 |

---

## 4 稳定性与版本

| 术语 | 定义 | 出处 |
|---|---|---|
| S0 | 冻结。永不破坏，只能追加。由语料库测试保证 | 03 文档第 2 节 |
| S1 | 稳定。语义不变，允许兼容扩展；破坏性变更需要 major 与迁移工具 | 03 文档第 2 节 |
| S2 | 实验。可以随时修改，需要显式开关 | 03 文档第 2 节 |
| S3 | 内部。不做承诺 | 03 文档第 2 节 |
| 语料库测试 | 用历史所有版本的包加载当前引擎，必须 100% 通过。S0 的可执行定义 | 03 文档第 2 节 |
| `struct_size` | 结构体的首字段，用于末尾追加字段而不破坏二进制兼容 | 04 文档第 3 节 |
| `arch_family` 与 `arch_feature` | 必须用二元组比较，不能按整数大小比较，因为 `sm_90a` 不等于 `sm_90` | 03 文档第 5.1 节 |

---

## 5 Kernel Harness 与 Agent

| 术语 | 定义 | 出处 |
|---|---|---|
| Kernel Harness | 用 Agent 编写 kernel 的离线系统，做语义分析并产出 plan 与 KernelUnit，不进入运行时 | 08 文档 |
| SemanticGraph | Kernel Harness 内部 IR，唯一允许携带语义的数据结构 | 08 文档第 3 节 |
| FusionDecision | 融合分组建议，由 Agent 产出，需通过 D1 到 D6 校验 | 08 文档第 5 节 |
| emit | 确定性合成器，唯一产出 plan 的地方，禁止调用 LLM | 08 文档第 7 节 |
| 布局传播 | 全局决定每个张量的 layout，使转换代价最小 | 08 文档第 6 节 |
| 协调者 | 唯一非 LLM 的角色，掌握所有状态转移权限 | 07 文档第 2 节 |
| 修复建议 | 对某类失败的固定下一步动作，不交给 LLM 决定 | 07 文档第 6 节 |
| debug bundle | 人工介入时交付的完整复现包 | 07 文档第 7.2 节 |
| provenance | 溯源：哪个模型、哪个 prompt、哪个 seed、谁评审 | 06 文档第 5.1 节 |

---

## 6 校验

| 术语 | 定义 | 出处 |
|---|---|---|
| golden | 参照输出，必须由 fp64 参照实现产生，否则真值本身是错的 | 09 文档第 5.2 节 |
| 误差预算标定 | 用 `fp16_ref` 与 `fp64_ref` 的差测出该 node 的固有误差，容差取它的若干倍 | 09 文档第 5.2 节 |
| L0 静态校验 | schema、连接性、符号完整性 | 09 文档第 5.1 节 |
| L1 差分校验 | 逐 node 对比 fp64 golden | 09 文档第 5.1 节 |
| L2 统计校验 | 形状扫描乘以输入分布扫描 | 09 文档第 5.1 节 |
| L3 任务级校验 | 端到端指标，如 mAP 或 top-1，唯一能证明整体没有错的门禁 | 09 文档第 5.1 节 |
| 依赖探针 | 扰动每个输入元素，检查输出必须有响应，用于抓漏算 | 09 文档第 5.3 节 |
| 对齐扰动 | 把 arena 基址偏移若干字节后重跑，抓隐式对齐假设 | 09 文档第 2.3 节 |
| canary | 在 buffer 两端插入哨兵区，检测越界写 | 09 文档第 2.4 节 |
| reward hacking | Agent 优化"通过检查"而不是"正确"，已有真实事故 | 14 文档第 1 节 |
| spatial_pattern | 错误元素的空间分布形态，用于推断 bug 类型 | 09 文档第 5.4 节 |

---

## 7 性能与诊断

| 术语 | 定义 | 出处 |
|---|---|---|
| 瓶颈分类 | 从耗时与声明的 flops、bytes 推断出访存受限、计算受限或延迟受限 | 09 文档第 6.2 节 |
| roofline 元数据 | manifest 中声明的 flops 与 bytes 表达式，使引擎无需 NCU 即可分类 | 09 文档第 6.2 节 |
| kernel 时间占比 | kernel 时间除以端到端时间，判断引擎开销是否失控的唯一指标 | 09 文档第 9 节 |
| 三级形状策略 | L0 符号化、L1 热形状特化、L2 形状类特化 | 05 文档第 5.1 节 |
| 降级顺序 | tuned、默认参数、其他 arch、ref、拒绝。每级必须可见 | 09 文档第 7.1 节 |

---

## 8 容易被混用的三组词

### 8.1 op、node 与 KernelUnit

- op：语义概念，例如卷积、加法。只在 Kernel Harness 中出现。
- node：plan 中的一次调用，它不知道自己算什么。
- KernelUnit：一个可执行文件，实现某个融合块。

一个 node 引用一个 KernelUnit；一个 KernelUnit 可以被多个模型、多个 node 复用。

### 8.2 编译、生成与执行编排

- 编译：有确定的、可复现的变换规则，例如 Kernel Harness 的 `emit`。
- 生成：需要搜索或创造，例如 Agent 写 kernel。
- 执行编排：把已有的 kernel 按显式 plan 拼起来，由引擎负责。

本项目把生成交给 Agent，把编译留在 Kernel Harness，把执行编排留给引擎。三者不混。

### 8.3 稳定性与重要性

一个东西很重要，不代表它应当进入 S0。

判据只有一个：如果它明天变了，引擎要不要改代码。要改，就不能进稳定层。

这也是为什么"融合策略"极其重要，但它是 S2；而"`SymExpr` 的除法语义"看起来琐碎，却是 S0。
