# 决策 0008 grid 与 block 由 KernelUnit 的 launcher 在调用时计算

- 状态：已接受
- 时间：2026-10

## 背景

kernel 的执行配置，包括 grid 维度、block 大小和动态 shared memory 大小，可以在两个地方决定：

- 在可执行格式里固定。编译期算好，写进 plan 或可执行文件。IREE 的 `hal.executable.export` 采用这种做法，TensorRT 也偏向它。
- 在调用时计算。由 kernel 自带的 host launcher 读取 dims 现算。

## 决定

采用第二种。引擎只传缓冲区与 stream，完全不参与 grid 与 block 的计算。

## 理由

这个取舍换来三件事。

第一，符号化 kernel 成为可能。一个 KernelUnit 读取 dims 在运行时算 grid，覆盖所有 shape，形状组合不会爆炸，见 05 文档第 5.1 节的 L0 层。

第二，引擎不需要知道 workgroup 数量，因此不需要为不同 shape 生成不同的可执行包。

第三，Agent 可以在 launcher 里做任意决策，例如 persistent kernel、split-K、动态 smem、多 kernel 流水和 warp specialization。引擎一概不知。

第三点尤其重要。这正是 TVM 无法表达 Hopper 的那部分自由度。本项目不需要表达它，只需要不挡住它。

## 后果

正面效果是动态形状的成本大幅降低，kernel 内部不受限制，ABI 极简。

负面效果有三点：每次调用要重算 grid，约 1 到 3 μs；无法做跨 kernel 的编译期优化，例如融合两个 kernel 的 grid；符号化 kernel 可能比特化版本慢 5% 到 15%。

## 与 IREE 的对比

IREE 的做法带来更好的编译期确定性，代价是形状变化就要重新编译。本项目选择调用期决定，因为推理场景里 1 到 3 μs 可以忽略，而形状组合的可控性更重要。

这是由场景驱动的取舍，不是绝对优劣。

## 相关文档

04 文档第 1 节 I7、14 文档第 3.1 节、12 文档第 2.1 节。
