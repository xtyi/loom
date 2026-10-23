# 08 Kernel Harness（loom-harness）

Kernel Harness 是用 Agent 编写 kernel 的那套离线系统：把模型转成 plan，驱动 Agent 产出 KernelUnit，跑校验与调优，最后打出包。

引擎不需要理解模型语义，Kernel Harness 必须理解，而且越完整越好。它可以依赖 PyTorch、ONNX、LLM，以及任何明年就会被淘汰的库，因为它不在推理路径上。

---

## 1 为什么放在引擎外

| 维度 | 运行时 | Kernel Harness |
|---|---|---|
| 运行环境 | 客户的生产机 | 开发机与 CI |
| 依赖 | 无 Python、无 ONNX、无编译器 | 可以有全部依赖 |
| 变化速度 | 年 | 周 |
| 失败后果 | 线上事故 | 重跑一次 |
| 是否需要语义 | 不能有 | 必须有 |
| 维护者 | 少量系统工程师 | Agent 加少量算法工程师 |

如果把 Kernel Harness 的能力塞进运行时，运行时会重新长成一个编译器，那正是 00 文档要避免的。

工程上的说法是：运行时是交付给客户的产品，需要稳定、体积小；Kernel Harness 是内部的构建工具，需要能快速修改，出问题重跑即可。

---

## 2 流水线与产物

```
  模型文件（第一版优先 ONNX）
       │  ingest
       ▼
  SemanticGraph ────────────────┐  Kernel Harness 内部 IR，含算子语义
       │  precision + refrun    │  引擎看不到它
       ▼                        │
  数值方案与基线参考            │
       │  partition ◄───────────┤  Agent：融合分组建议
       ▼                        │
  FusionDecision                │
       │  layout                │
       ▼                        │
  LayoutDecision                │
       │  按最终融合/数值边界刷新参考
       │  emit ◄────────────────┘  确定性合成器，唯一产出 plan 的地方
       ▼
  plans/*.json 与 mem/*.json
       │  gen ◄── Agent：写 kernel 实现
       ▼
  kernels/<ku>/{ku.json, src/, lib/}
       │  check，调用引擎的校验服务
       ▼
  ConformanceReport ──┐
       │  tune         │ 失败则回到 gen，带报告
       ▼               │
  bench/report.json ◄──┘
       │  pack
       ▼
  model.loom ── 注册
```

每个箭头都是可以重跑、可以 diff、可以单元测试的步骤。中间产物全部落盘，可以从任意一步重新开始，这对 Agent 闭环很重要，不必每次都从头生成。

---

## 3 SemanticGraph

这是唯一允许携带语义的数据结构，因此它被严格限制在 Kernel Harness 内部。

```python
@dataclass
class SNode:
    id: int
    op: str                     # "conv" | "add" | "concat" | ...
    attrs: dict
    inputs: list[TensorRef]
    outputs: list[TensorSpec]

@dataclass
class TensorSpec:
    name: str
    dtype: str
    shape: list[str | int]      # "N" 表示符号维度
    layout: str                 # "nhwc" | "nchw" | "blocked32"
```

三条规则：

1. `SemanticGraph` 只存在于 `harness/` 目录下。`runtime/` 与 `cli/` 不得 import 它的类型，CI 检查。
2. 它必须带完整的 shape 与 layout 信息，否则 `emit` 无法工作。
3. 它是不可变的，所有变换产出新图，便于 diff 与回溯。

它不是 Relay 或 Relax 那样的通用 IR。它只支持目标模型族（CNN 与检测网络），不追求表达 attention、稀疏或控制流。一旦追求通用，就会重新长成编译器。

---

## 4 精度规划与参照执行

### 4.1 PrecisionDecision

Kernel Harness 决定计算方案，runtime 执行已经决定的方案。第一版从模型原有 dtype 起步，允许显式指定 FP32/FP16 混合方案；BF16、自动混合精度搜索和 INT8/FP8 校准按需求追加。

构建产物 `PrecisionDecision` 应包含：图中值的存储类型、各 KernelUnit 的数值契约、转换边界、模型 I/O 类型、必要的量化参数绑定，以及应用方给出的质量预算。存储 dtype 与内部计算/累加类型分开记录。融合、布局和精度会相互影响，可以迭代，最终 plan 必须重新验证。

固定权重转换优先离线；Cast、Quantize、Dequantize 和动态 scale 计算可成为普通节点，或被融合到声明了相应数值契约的实现里。不能由 runtime 因为看到某个 op 名称而隐式改变精度。量化不仅是换 dtype，还需 scale、zero point、轴/分组和舍入/饱和规则（按方案适用）。

可以复用外部转换与校准工具，先验证转换后的图，再生成 KernelUnit。现有方案示例见 [ONNX Runtime 浮点转换](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html) 与 [量化说明](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)。

### 4.2 两套参考用途

原始模型基线用于模型质量评估；符合部署数值契约的独立参考用于 kernel 正确性检查。记录两者的版本、输入、权重和数值设置，避免共模错误。FP64 仅在适用时用于数值分析，不是强制的通用 oracle；详细规则见 09 文档第 5.2 节。

按最终融合边界捕获参考输入输出；局部差分使用相同输入隔离 kernel 错误，完整 plan 的端到端检查单独衡量累积误差。参考数据按其真实 dtype 保存，转换存储精度需明确记录并验证。中间结果可按需生成，不要求把所有节点的 FP64 数据全量存盘。

留出数据与发布任务指标由独立验证流程管理，Agent 不得更改。在线默认不加载 golden，按需校验时才使用。

---

## 5 融合决策

### 5.1 决策格式

```json
{
  "groups": [
    {"id": "g17", "nodes": [12,13,14,15],
     "rationale": "C2f：split 加三个 conv 加 concat 共用同一批特征图，融合可省三次写回",
     "est_gain": "medium", "risk": "high"},
    {"id": "g21", "nodes": [18], "rationale": "孤立 conv，保留为厂商库调用"}
  ],
  "constraints_seen": {"max_smem_kb": 228}
}
```

### 5.2 结构校验

Agent 的分组建议先过一遍纯结构检查，不合法直接打回，避免浪费一次 kernel 生成。

```
D1 覆盖性：每个 node 恰好属于一个 group
D2 连通性：每个 group 的 node 集合在数据流上连通
D3 边约束：group 内部不允许有边绕出去再回来
D4 规模：group 的 node 数不超过 MAX_FUSE，默认 8
D5 类型：group 内全为可融合的类型组合
D6 输出：group 对外输出数不超过 MAX_OUT，默认 4
```

D5 需要语义，但它在 Kernel Harness 里，这是允许的。这些检查本身属于 01 文档 P15 的具体体现。

### 5.3 三种策略

| 策略 | 实现 | 使用阶段 |
|---|---|---|
| `rules` | 基于算子类型与形状的固定规则，例如 conv 加 bn 加 act 必融 | 第一版起步 |
| `agent` | LLM 提分组建议，经 D1 到 D6 校验 | 后续 |
| `search` | 枚举候选分组，用成本模型打分 | 更后续 |

即使采用 `agent` 策略，也必须有 `rules` 兜底。分组决策的搜索空间虽然大，但大部分收益来自少数几个明显模式：conv-bn-act、split-concat、upsample-concat。规则能拿到其中大部分，Agent 负责剩下的难题。

这也回答了"Agent 漏掉融合机会"这一风险：规则先拿到基础收益，Agent 只在规则之上的增量部分工作。

---

## 6 布局传播

一张四维张量可以表达成 NCHW、NHWC、blocked-32、blocked-8 以及各种 pad 后的变体。做融合之后，布局选择变成一个全局约束问题：

- 一个 fused conv 内部用 NHWC 通常更快。
- concat 沿通道方向拼接，NHWC 下是连续的，NCHW 下不连续。
- 某些 vendor kernel 只接受 NCHW。

解法是在 Kernel Harness 里做全局传播：

```python
def propagate_layout(graph, anchors):
    """anchors 是必须固定的布局：vendor kernel 的强制要求、I/O 格式。
       其余节点在候选布局中选择，目标是布局转换次数加估计转换代价最小。"""
```

这是一个图着色或最小割问题，贪心加局部搜索即可。关键在于它在 Kernel Harness 中，因此可以知道"concat 沿哪个轴"这类语义。

引擎看到的结果只是每个 slot 有一个 `layout` 整数和一组 `stride`，它不知道 NHWC 是什么。

布局转换作为一个普通 node 插入 plan，使用一个 transpose KernelUnit。不存在隐式布局转换：如果 plan 里没有转换节点，说明布局是一致的，引擎不需要也不能检查语义一致性。

---

## 7 emit：确定性合成器

`emit/` 是整个项目中唯一产出 plan 的地方，也是唯一要求零 bug 的地方。它不允许调用 LLM，输出必须能逐字节复现，改动必须伴随 plan 的 diff 评审。

### 7.1 输入与输出

```
输入：
  SemanticGraph，已应用 PrecisionDecision、FusionDecision 与 LayoutDecision
  已验证的形状/类型约束；golden 只覆盖样本，不能替代动态 shape 约束
  各组的 kernel manifest，提供 io 声明
  shape 的符号映射策略

输出：
  plans/main.json        symbols、domain、slots、nodes、variants、mem
  plans/mem/*.json       显式内存计划
  打包后的 params 字节
  每个 node 的 derived 表达式
```

### 7.2 七步算法

每一步都是纯函数。

```
1. 分配 slot
   对 SemanticGraph 的每个张量值（按"值"而不是按 op）分配恰好一个 slot。
   不变量 T1：每个张量值恰好一个 slot，无遗漏无重复。
   融合组内部的中间值不分配 slot，它们在 kernel 内部。
   融合组的外部输入输出分配 slot。

2. 计算 shape 表达式
   把 semantic shape 转成 SymExpr 字符串。
   整数写字面量，符号写名字，派生维度按算子语义展开，
   例如 concat 的通道维写成 "C1+C2"。

3. 计算 stride
   按 layout 计算紧凑布局；有 pad 时，最后一维 stride 仍为 1，
   行 stride 为 align(C, 32)。
   不变量 T2：每个 slot 的 stride 自洽，单射且单调。

4. 生成 node
   每个融合组对应一个 node。
   in/out 按 manifest.io 的顺序排列。
   不变量 T3：数量与顺序必须与 manifest 完全一致，
   由读取同一份 manifest 保证，而不是靠人工对齐。

5. 生成 derived 表达式
   从 manifest 声明的 derived 名字反查它在语义图上的含义。

6. 打包 params
   按 manifest.params 的 schema，从 FusionDecision 与调优结果取值。

7. 生成内存计划，explicit 模式
   先确定合法 schedule，再依据完成关系计算存活性，做全局分配，可以用 best-fit 或颜色分配，
   比引擎的 first-fit 更好。
   不变量 T4：结果必须通过引擎的验证规则，在 emit 内部先跑一遍。
```

### 7.3 三条工程纪律

1. `emit(graph, decision)` 是纯函数，相同输入必须产出逐字节相同的输出。测试方式是跑两次并比较字节。
2. 结束前用与引擎相同的验证器跑一遍，不通过就抛异常，不产出 plan。
3. `emit` 的任何改动，PR 中必须展示 plan 的 diff。plan 是最终产物，它的 diff 比代码 diff 更能暴露问题。

### 7.4 为什么不让 Agent 生成 plan

一个 YOLO plan 有约 200 个 slot、40 个 node、600 多个数字。让 LLM 生成这个的出错率接近 100%，而且错误是静默的：连线错了不会报错，只会算错。

生成过程不需要创造力，它是图上的机械变换。一旦出错，调试成本是人工方案的几十倍。

判断标准很简单：如果 `emit/` 的代码里出现对 LLM 的调用，CI 直接失败。

---

## 8 gen：驱动 Agent 写 kernel

### 8.1 脚手架

```bash
loom-harness gen --work work/ --group g17
```

生成：

```
work/groups/g17/
├── ku.json                 manifest 模板，io 部分已填好
├── README.md               任务说明，含验收标准
├── src/
│   ├── impl.cu             Agent 要填的空壳，含 lm_ku_entry_v1 签名
│   ├── ref.cpp             参考实现，从 PyTorch 导出，Agent 不修改
│   └── shim.cpp            使用 Triton 或 TileLang 时的 launcher 包装，模板生成
├── build.sh                确定的编译命令
├── cases/
│   ├── n1_640/inputs/*.bin  outputs/*.bin
│   └── n4_640/...
└── check.sh                一键执行 build、跑 cases、打印结构化报告
```

`check.sh` 是 Agent 的"编译器"。它输出的不是错误，而是 09 文档第 5.4 节定义的结构化报告。Agent 的迭代速度取决于反馈质量，不取决于模型能力。

### 8.2 给 Agent 的任务说明

```markdown
## 你的任务
实现 ku=c2f_block 的 fast 实现，替换 src/impl.cu 中的 TODO 部分。
输入：(N,H,W,C) f16 与 (C,C1) f16 权重
输出：(N,H,W,C2) f16
数学定义见 ref.cpp，可读但不要修改。

## 你不能修改
- lm_ku_entry_v1 的签名与语义
- ku.json 的 io 声明
- src/ref.cpp
- build.sh 的编译选项，-O 与 -arch 相关除外

## 验收标准
1. check.sh --level L1 全部通过，比较规则由独立验证流程确定，Agent 不得修改
2. check.sh --level L2 通过，形状扫描
3. bench.sh 报告的耗时不超过给定阈值
4. workspace 不超过 100 KB
5. 必须声明 deterministic

## 可用资源
- 参考实现：src/ref.cpp
- 相似已实现 kernel：kernels/conv_bn_silu/src/impl.cu
- 硬件：sm_90，228 KB smem，132 个 SM
```

### 8.3 模板与 shim

Agent 不应该手写样板部分：参数解包、grid 计算、错误码、caps 导出。这些由模板生成，Agent 只写 kernel body。

```
templates/
  cuda_cpp/{elementwise, matmul, conv, fused_block}/
  triton/...
  tilelang/...
```

这是 01 文档 P5 在 kernel 层的落地：结构固定，body 由 Agent 生成。

### 8.4 默认方言

| 方言 | 定位 | 理由 |
|---|---|---|
| TileLang | 默认 | 显式管理 shared、fragment、TMEM 与 pipeline；编译错误带 Python 行号；可内嵌原始 CUDA 作为逃生舱；CUDA 路径覆盖 SM70 到 SM120 |
| Triton | 快速原型 | 代码最短，适合先探索算法结构；但 tensor core 细节由编译器决定，出错难定位 |
| CUTLASS 与 CuTe | 性能攻坚 | 前两者都到不了 roofline 时使用 |
| CUDA C++ | 兜底 | 最灵活，但生成质量方差最大 |
| vendor | 优先 | 适用时直接用，见 06 文档第 7 节 |

对 Agent 而言，"代码短"不如"错误信息可定位"，这是把 TileLang 而非 Triton 放在默认位置的主要理由，与人类开发者的偏好相反。

---

## 9 check、tune、pack、bench

| 命令 | 做什么 | 输出 |
|---|---|---|
| `loom-harness check --level L1` | 调用引擎的 `lm_validate` | `ConformanceReport` |
| `loom-harness check --level L2` | 形状与输入分布扫描 | 同上，含失败形状列表 |
| `loom-harness tune` | 枚举参数组合，实测选最优 | 最优参数与扫描曲线 |
| `loom-harness bench` | 端到端基准，固定环境 | `bench/report.json` |
| `loom-harness pack` | 组装 `.loom`，签名，许可扫描 | 包文件 |
| `loom-cli validate --pkg` | 对已打包产物做校验，与 check 同源 | 报告 |

调优的纪律：

- 搜索空间必须有限且提前声明，不允许无界搜索。
- 每次测量要有 warmup，并多次取中位数，GPU 测量噪声较大。
- 记录测量噪声，即多次测量的标准差。如果最优与次优的差距小于噪声，选参数更保守的那个。

---

## 10 命令行

```bash
# 全流程
loom-harness build --model yolov8n.onnx --target sm90 --cases cases.json --out work/
loom-harness pack  --work work/ --out yolov8n-h100.loom

# 分步，供 Agent 闭环使用
loom-harness refrun     --work work/ --case n1_640
loom-harness partition  --work work/ --strategy rules
loom-harness emit       --work work/ --out plans/
loom-harness gen        --work work/ --group 17
loom-harness check      --work work/ --group 17 --level L1
loom-harness tune       --work work/ --group 17 --budget 300s
loom-harness bench      --work work/ --vs trt,ort

# 部署侧
loom-cli run      --pkg yolov8n-h100.loom --input in.bin --iters 100
loom-cli validate --pkg yolov8n-h100.loom --case n1_640
loom-cli profile  --pkg yolov8n-h100.loom --input in.bin --out trace.json
loom-cli inspect  --pkg yolov8n-h100.loom
```

---

## 11 质量要求

Kernel Harness 是构建工具，它的 bug 会被复制到每一个包里。

| 要求 | 检查方式 |
|---|---|
| `emit` 是纯函数 | 相同输入跑两次，输出字节相同 |
| 每个模块有单元测试 | 覆盖率不低于 70% |
| 中间产物可 diff | JSON 格式化稳定，键排序 |
| 端到端可重跑 | 可以从 work/ 的任意阶段继续 |
| 依赖固定 | `requirements.lock`，工具版本写入 provenance |
| 无隐式全局状态 | 每个命令是纯函数加文件 I/O |

---

## 12 常见问题

| 做法 | 后果 |
|---|---|
| 让 LLM 生成 plan | 静默的连线错误，调试成本极高 |
| Kernel Harness 的类型出现在运行时接口里 | 运行时被迫依赖 Python 和语义 |
| golden 用 fp16 计算 | 校验失去意义 |
| 融合决策不做结构校验就生成 | 浪费 GPU 时间，Agent 收到无效反馈 |
| 布局转换藏在 kernel 内部 | 布局决策不可见、不可优化、不可校验 |
| `emit` 中写特殊情况分支 | 每个特判都是一个未来的 bug |
| 只测一个 shape | Agent 学会过拟合 |
| 调优搜索无界 | 时间失控，结果不可复现 |
