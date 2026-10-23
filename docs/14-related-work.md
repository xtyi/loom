# 14 相关工作与定位

本文回答两个问题：别人已经做到哪一步，我们凭什么不重复造轮子。

所有结论都附来源。这一章的作用是避免在"没人做过"的假设上重复别人的问题。

---

## 1 Agent 生成 kernel 的真实现状

这一节直接决定 06 文档第 7 节（厂商库优先）和 12 文档的风险判断是否成立。

| 工作 | 关键数据 | 对本项目的含义 |
|---|---|---|
| KernelBench，250 个 PyTorch 任务，分 L1 到 L4 | 主指标是 `fast_p`，即输出正确且相对 PyTorch Eager 加速不低于 p 的比例。原论文中前沿模型仅不到 20% 的用例追平 baseline | "能跑对"和"达到库级"是两件事 |
| KernelBench-Verified，把 baseline 换成开启 TF32 的 PyTorch，并增加四组隐藏输入分布 | 最好的单轮几何平均只有 0.88 倍，原协议下报告的是 1.43 倍。没有模型能稳定超过 PyTorch，且 28% 的 kernel 反而抬高了峰值显存 | baseline 必须是最强库而不是 eager；必须核算峰值显存；隐藏分布是有效手段 |
| Kevin，QwQ-32B 的多轮强化学习 | L1 与 L2 正确率从 56% 升到 82%，平均加速从 0.53 倍升到 1.10 倍，对手是 PyTorch Eager，硬件 H200，优于 o4-mini 的 0.78 倍 | 多轮 refine 明显有效，但 1.10 倍是对 eager，不是对 cuBLAS 或 CUTLASS |
| AlphaEvolve | 为 4×4 复数矩阵乘找到 48 次标量乘法，是 56 年来首次改进 Strassen；优化了 Gemini 训练用的 matmul 与 attention kernel；不是通用 kernel 生成器 | 进化式搜索在特定问题上能超过人类，但需要可验证的目标函数和大量算力 |
| Sakana AI CUDA Engineer | 发布时宣称最高 150 倍，社区复现发现该例实际慢 3 倍；作者承认模型绕过了正确性校验；后续转向 robust-kbench，用 LLM verifier 过滤作弊解 | reward hacking 已经发生过真实事故，不是理论担忧 |
| GPU MODE 竞赛 | 一个 Codex 自动研究循环 14 天提交 1500 多次，在 B200 QR 赛中 183 人里排第 12，相对 baseline 232 倍 | 在目标明确、可自动评测、可大量试错的赛道上，Agent 已经很有效 |
| TVM 与 MetaSchedule | AutoTVM、Ansor、MetaSchedule 三代演进，2025 年维护者确认改用 MetaSchedule，TVM 本身仍活跃，已发布 v0.26 | "TVM 的抽象被击穿"不等于"TVM 项目已死"，批评应当针对抽象层次的选择，而不是项目活跃度 |

### 1.1 三条对设计有直接影响的结论

第一，Agent 打不过调优库，至少现在打不过。所有"超过 baseline"的数据，baseline 都是 PyTorch Eager 或未调优实现；对上 cuBLAS、CUTLASS、TensorRT，公开数据普遍低于 1 倍。

由此产生的设计后果是：`kind: vendor` 必须是一等公民（06 文档第 7 节），Agent 的首要职责是选择（01 文档 P16）。

第二，reward hacking 已经发生过真实事故。Sakana 的案例说明，如果验收只看"跑通且变快"，Agent 会复制参照实现、硬编码测试值或跳过计算。

由此产生的设计后果是：容差必须由 fp64 golden 标定而不是人为给定（09 文档第 5.2 节），并加入 Agent 看不到的检查维度（09 文档第 5.3 节）。

第三，多轮 refine 加固定预算明显优于单轮。Kevin 的 16 路乘以 8 轮串行优于 128 路单轮。

由此产生的设计后果是：协议必须内建尝试预算与结构化反馈（07 文档第 6 节），而不是一次生成、一次验收。

---

## 2 Kernel DSL 版图

| DSL | 抽象层级 | Hopper 与 Blackwell 支持 | 对 Agent 的友好度 |
|---|---|---|---|
| Triton | block 级 Python DSL | Hopper 通过 TMA 与 WGMMA；Blackwell 的 tcgen05 由 Gluon 层暴露 | 代码最短，但 tensor core 细节由编译器决定，出错难定位 |
| TileLang | 架在 TVM 上的 tile DSL，显式分配 shared、fragment、TMEM 并控制 pipeline | 覆盖 SM70 到 SM120，支持 TMA、WGMMA、TCGEN5，可用 `T.CUDASourceCodeKernel` 内嵌原始 CUDA | 编译错误带 Python 行号，控制粒度细，更适合 Agent 迭代 |
| CUTLASS 与 CuTe DSL | C++ 模板库加 Python CuTe DSL | SM80 及以上，有 wgmma 与 tcgen05 专门指南 | 性能上限最高，但学习与生成成本高，适合最后一公里 |
| cuTile | NVIDIA 新的 tile DSL | 仅支持 Blackwell，需要 CUDA 13.1 | 太新，覆盖太窄，第一版不用 |
| Mojo 与 MAX | Python 超集 | sm_75 到 sm_120 | 生态尚未成熟，暂不押注 |
| Pallas | 直接暴露 wgmma、tcgen05、TMEM | Ampere、Hopper、Blackwell | 官方明示 API 高度不稳定 |
| ThunderKittens | header-only C++ tile 原语 | 2.0 支持 Blackwell 的 TCGEN05、MXFP8、NVFP4 | 适合作为高效能模板库，而不是生成目标 |

结论是：默认方言用 TileLang，Triton 用于快速原型，CUTLASS 与 CuTe 用于性能攻坚。

对 Agent 而言，"代码短"不如"错误信息可定位"。这是把 TileLang 放在默认位置的主要理由，与人类开发者的偏好相反。

---

## 3 runtime 与编译分离的先例

| 系统 | 可执行格式 | kernel 接入方式 | 编译发生在哪里 | 借鉴与差异 |
|---|---|---|---|---|
| ONNX Runtime | `.onnx` 加 execution provider | provider 以 C ABI 插件库接入，但 kernel 注册是 C++ ABI，必须与 ONNX Runtime 版本对齐 | 图优化在 session 初始化时 | 借插件库形态；差异是本项目坚持纯 C ABI，避免版本对齐痛苦 |
| IREE | `.vmfb` | `iree_hal_executable_plugin`，带版本查询的 dlopen ABI，按 base 加 offset 传参 | 编译期决定一切 | 借可执行包加插件 ABI；关键差异见第 3.1 节 |
| TVM | `VMExecutable`，含 bytecode、常量池与 kernel | 统一走 `PackedFunc`，可以从 imported module 或全局 registry 解析 | 编译期 | 借 plan 中 Call 指向外部符号的形态；差异是不做通用 IR |
| Modular MAX | MEF，序列化的图编译产物 | C API 加载执行，支持 CUDA Graph capture | 编译期 | 借脱离 Python 的 C API；差异是 MEF 设备相关且只能在相似系统间迁移，正是本项目要用指纹显式解决的问题 |
| TensorRT | `.engine` 或 plan 文件 | `IPluginV3::enqueue(inputs, outputs, workspace, stream)` | 构建期，含 tactic 选择 | 借 plan 文件加 runtime 分离的形态；差异是 TRT 的 plan 记录版本与 compute capability，跨架构默认反序列化失败，本项目把这件事显式化为指纹 |
| tinygrad | 无 | UOp 到 kernel 的 JIT 编译器加极小 runtime | 运行期 JIT | 不是本项目要的形态，它把编译放进了 runtime |
| ggml 与 llama.cpp | 无 plan 格式 | 各后端手写 kernel，纯 C，运行时零分配 | 手写 | 借 runtime 优先与零分配的纪律；差异是没有 plan 格式，无法被 Agent 驱动的工具链复用 |

### 3.1 与 IREE 的关键差异

IREE 的 `hal.executable.export` 在编译期固定了 workgroup size 与 count、subgroup size 和 pipeline layout。

直接后果是形状一变就要重新编译。这是它在动态形状下工作量的来源，也是"动态形状把你逼回编译器"这一质疑的由来。

本项目的 ABI 刻意不这么做：引擎只传缓冲区与 stream，grid 与 block 由 KernelUnit 的 launcher 在调用时计算（04 文档第 1 节，决策 0008）。

这个分工差异带来三个收益：

1. 符号化 kernel 成为可能，一个 KernelUnit 读取 dims 在运行时算 grid，覆盖所有形状，形状组合不会爆炸。
2. 引擎不需要知道 workgroup 数量，因此不需要为不同形状生成不同可执行包。
3. Agent 可以在 launcher 里做任意决策，例如 persistent kernel、split-K、动态 smem，引擎一概不知。

代价是 launcher 每次调用要重算 grid，约 1 到 3 μs。在推理场景可以忽略，在训练的高频小 kernel 场景不可忽略，但这不是本项目的目标。

这是整个方案里最有技术含量的一处取舍：把编译期决定换成调用期决定，用少量运行时开销换取形状组合的可控性。

### 3.2 是否有人做过同样的事

没有项目同时把"plan 格式加 kernel 插件 ABI"做成一等公民，并且把算子生成完全外置给外部 Agent。

最接近的三个：

- IREE：格式与插件 ABI 都有，但图和调度仍由编译器决定，插件只能填空。
- TVM Relax VM：`VMExecutable` 加 `PackedFunc` 的形态很接近，但整个系统的价值主张仍绑在"编译器能生成好 kernel"上。
- TensorRT：plan 加 `IPluginV3` 的产品形态最接近，但 plan 由 TRT 的构建器产出，用户无法让外部 Agent 接管 kernel 生成。

本项目的差异只有一点，但足够根本：它们假设程序由编译器产生，本项目假设程序由 Agent 产生。前者需要编译器，后者只需要接口约定。而恰好在这几年，后者是更便宜、更稳定、也更长寿的那一半。

---

## 4 可借鉴的 ABI 与格式设计

| 来源 | 借鉴点 | 落在本文档集的哪里 |
|---|---|---|
| TensorRT 的 `IPluginV3::enqueue` | "指针数组加 workspace 加 stream"是事实标准 | 04 文档的 `LmKernelArgs`，本项目只是把它形式化并冻结 |
| IREE executable plugin | ABI 带版本查询，失败可诊断 | `lm_ku_caps_v1` 与加载协议 |
| IREE 的 base 加 offset 传参 | 避免指针重定位 | 用 `LmBuffer.data` 加 `elem_offset` 兼顾两者 |
| ONNX Runtime custom op 的版本对齐问题 | 反面教材 | 坚持纯 C ABI，不做 C++ ABI 耦合 |
| TensorRT 的 plan 绑定版本与 compute capability | 可移植性应当显式声明 | fingerprint、requires、coverage |
| MAX 的 MEF 设备相关 | 同上 | 同上，并区分 strict 与 lenient，标记 unvalidated |
| TVM 的 `PackedFunc` registry | plan 中的 Call 指向外部符号 | plan 的 `node.ku` 与动态库符号解析 |

### 4.1 CUDA Graph 的已知限制

| 事实 | 对应处理 |
|---|---|
| 捕获期间禁止同步与查询，禁止 legacy stream，禁止 `cudaMemcpy` 这类同步接口 | kernel 约定禁止内部同步，只用异步接口 |
| 捕获把 kernel 参数按值固化，包括指针 | arena 偏移必须确定，graph key 必须包含决定偏移的全部信息 |
| 动态分配需要 graph memory node | 一次性预分配 arena，不做动态分配 |
| 动态控制流需要 12.4 以上的 IF 与 WHILE，或 12.8 的 SWITCH | 第一版不支持控制流，NMS 用固定容量加 count 表达 |
| 动态参数可用 `cudaGraphExecKernelNodeSetParams` 更新节点 | 后续优化方向：跨形状复用同一 graph 结构，只更新参数 |
| 形状变化不适合直接放进 graph | 每个形状一个 graph，配 LRU |

---

## 5 定位

已有的系统都在把图变成程序上投入巨大：TVM 想用 IR 加 schedule 表达计算，TensorRT 想用 tactic 搜索吞掉 kernel 选择，IREE 想用编译期固定一切来换确定性。

它们的共同假设是程序必须由编译器产生。这个假设在 Hopper 之后开始失效，因为抽象跟不上硬件；在 Agent 出现之后开始变得不必要。

本项目假设程序由 Agent 产生。于是需要的基础设施从编译器变成了接口约定加运行时校验。

这不是"没有编译器"，而是把编译器搬到离线，并把它的输出标准化为 plan 与 KernelUnit。收益是接口约定比编译器寿命长；代价是每个模型与硬件组合都需要一次性投入，必须靠 Kernel Harness 与复用来压低边际成本。

---

## 6 参考文献

### Agent 与 kernel 生成

- KernelBench — https://arxiv.org/abs/2502.10517
- KernelBench-Verified — https://arxiv.org/abs/2607.16241
- Kevin — https://arxiv.org/abs/2507.11948
- AlphaEvolve — https://arxiv.org/abs/2506.13131
- AlphaEvolve 官方介绍 — https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/
- Sakana AI CUDA Engineer 的复现与更正 — https://gigazine.net/gsc_news/en/20250225-sakana-ai-cuda-engineer-walks-back
- robust-kbench — https://arxiv.org/abs/2509.14279
- GPU MODE 自动研究记录 — https://sankalp.bearblog.dev/autoresearch/
- GPU MODE 榜单 — https://www.gpumode.com/leaderboard/774
- TVM MetaSchedule 讨论 — https://discuss.tvm.apache.org/t/does-tvm-no-longer-use-autoscheduler/18041/2
- TVM v0.26.0 — https://github.com/apache/tvm/releases/tag/v0.26.0

### Kernel DSL

- Triton — https://github.com/triton-lang/triton
- Triton Gluon — https://triton-lang.org/main/gluon/api/nvidia.html
- TileLang — https://github.com/tile-ai/tilelang
- CUTLASS 与 CuTe DSL — https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/faqs.html
- cuTile — https://developer.nvidia.com/blog/how-to-write-high-performance-matrix-multiply-in-nvidia-cuda-tile/
- Mojo 硬件支持 — https://max.modular.com/gpu/hardware-compatibility/
- Pallas Mosaic GPU — https://docs.jax.dev/en/latest/jax.experimental.pallas.mosaic_gpu.html
- ThunderKittens — https://github.com/HazyResearch/ThunderKittens

### 运行时与可执行格式

- ONNX Runtime custom op — https://onnxruntime.ai/docs/reference/operators/add-custom-op.html
- ONNX Runtime plugin EP — https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries/development.html
- IREE HAL export — https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Dialect/HAL/IR/HALOps.td
- IREE executable plugin 示例 — https://github.com/iree-org/iree/blob/main/samples/custom_dispatch/cpu/plugin/README.md
- TVM Relax VM — https://tvm.apache.org/docs/_sources/arch/relax_vm.rst
- TVM PackedFunc 与 runtime — https://tvm.apache.org/docs/arch/runtime.html
- Modular MAX C API — https://github.com/modular/modular/tree/main/max/examples/capi
- TensorRT engine 兼容性 — https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html
- TensorRT plugin API 迁移 — https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/plugins-api-migration.html
- tinygrad runtime — https://github.com/tinygrad/tinygrad/blob/master/docs/developer/runtime.md
- ggml — https://github.com/ggml-org/ggml

### CUDA Graph

- CUDA Graph 指南 — https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html
- 动态参数 — https://developer.nvidia.com/blog/constructing-cuda-graphs-with-dynamic-parameters/
- 条件节点 — https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/
