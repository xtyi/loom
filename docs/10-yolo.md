# 10 首个落地目标：YOLOv8n

选一个具体模型把整条链路跑通，比在抽象层面争论有效得多。

本文给出 YOLOv8n 的落地方案，包括它为什么适合作为第一个目标，以及为什么它不适合作为商业场景。两点都需要说清楚。

---

## 1 为什么选 YOLO 做技术验证

| 理由 | 说明 |
|---|---|
| 结构固定且已知 | 可以做"手工基线对齐 Agent 产物"的对照实验，这是全项目最关键的证据 |
| 有客观指标 | mAP 可以自动计算，不依赖主观判断 |
| 有强基线 | TensorRT、ONNX Runtime、PyTorch 都是现成对照，跑一次就有数字 |
| 融合空间真实存在 | C2f、SPPF、upsample 加 concat 都是没有库提供现成 kernel 的融合体 |
| 前后处理复杂 | letterbox、DFL decode、NMS 覆盖动态输出、host 与 device 交互、固定容量等 ABI 难点 |
| 多 shape 真实 | 视频流的分辨率与 batch 都在变化，能压测动态形状机制 |
| 可拆解 | 三十多个 KernelUnit，可以分批替换，风险可控 |

---

## 2 为什么 YOLO 不适合作为商业场景

| 事实 | 后果 |
|---|---|
| 检测模型算子不到二十种，TensorRT 的整图融合加 tactic 选择已经做到极致 | 我们能拿到的增量有限 |
| 真实瓶颈常在后处理与多路调度，而不是卷积 | 优化卷积的边际收益低 |
| 大部分资本与人才流向 LLM | 逆流选方向，招人与融资都困难 |
| 性能领先 10% 很难证明价值，落后则立刻出局 | 风险收益不对称 |

所以定位要精确：YOLO 用来证明机制成立，不用来证明商业成立。

商业方向的候选在 11 文档第 8 节讨论，包括边缘设备上的小众模型族、需要长期维护的内部模型，以及要求可审计与可复现的受监管场景。

---

## 3 计算清单

YOLOv8n 在 640×640 输入下约为 3.2M 参数、8.7 GFLOPs。具体数值以本机实测为准。

```
输入 640×640×3 uint8
  │
  ├─ pre：letterbox、除以 255、HWC 转 CHW    可整体做成一个 KernelUnit
  │
  ├─ Conv k3s2 3→16 + BN + SiLU
  ├─ Conv k3s2 16→32 + BN + SiLU
  ├─ C2f 32→32（n=1）         融合体：split、两个 conv、concat
  ├─ Conv k3s2 32→64
  ├─ C2f 64→64（n=2）
  ├─ Conv k3s2 64→128
  ├─ C2f 128→128（n=2）
  ├─ Conv k3s2 128→256
  ├─ C2f 256→256（n=1）
  ├─ SPPF 256→256             融合体：三次 maxpool 加 concat
  │
  ├─ upsample 加 concat 加 C2f 512→128（n=1）
  ├─ upsample 加 concat 加 C2f 256→64（n=1）
  ├─ Conv k3s2 64→64 + concat + C2f 128→128（n=2）
  ├─ Conv k3s2 128→128 + concat + C2f 256→256（n=1）
  │
  ├─ detect：三个尺度，每个尺度两次 Conv k3，再接两个 1×1 conv 分支
  │
  └─ post：DFL decode、sigmoid、topK、NMS    输出长度可变
```

原始算子数约 120 到 150，取决于 BN 与 SiLU 是否单独计数。融合后约 30 到 35 个 KernelUnit。

这个四到五倍的削减，是端到端性能的主要来源之一，减少的是 launch 次数与中间张量写回。

---

## 4 融合分组

08 文档第 5 节的分组格式在 YOLO 上的具体形态：

| 组 | 内容 | 实现方式 | 预期收益 | 风险 |
|---|---|---|---|---|
| g00 | pre：letterbox、归一化、转置 | 自写，访存密集且无库可用 | 省两次全图拷贝 | 低 |
| g01–g10 | `Conv k3sN + BN + SiLU` | 厂商库（cuDNN）或自写 | 孤立卷积上库更强，见 01 文档 P16 | 低 |
| g11–g18 | `C2f` 整块 | 自写，无库可用 | 省三次写回与三次读 | 高 |
| g19 | `SPPF` | 自写，maxpool 链加 concat | 省三次写回 | 中 |
| g20–g21 | `upsample + concat` | 自写 | 省一次写回 | 低 |
| g22–g23 | `Conv s2 + concat + C2f` | 自写，跨层融合 | 收益大 | 高 |
| g24–g26 | detect head 的双卷积 | 自写，顺带融合 1×1 | 省四次写回 | 中 |
| g27 | `DFL decode + topK + NMS` | 自写，放在 device 端 | 省一次 D2H 回读 | 中 |

真正需要 Agent 的只有 g11 到 g18 和 g22 到 g23，共八到十个难组。其余用厂商库或简单 kernel 即可。

这就是 P16 的实战含义：把大部分算力交给库，把库做不了的部分交给 Agent。Agent 的工作量从三十个 kernel 降到十个，成本模型至少改善三倍。

---

## 5 三个 ABI 难点

### 5.1 可变长度输出

NMS 输出的框数量取决于输入内容，而 ABI 使用静态缓冲区。

解法是固定容量加计数，不引入可变长 buffer：

```
boxes   : f16 [max_det, 4]      max_det = 300，编译期常量
scores  : f16 [max_det]
classes : i32 [max_det]
count   : i32 [1]               实际有效数量
```

`count` 是一个 `role: scalar` 的 slot，位于 device 内存，需要时由 host 侧做一次 4 字节的 D2H 回读。消费方只读前 `count` 个。超过 `max_det` 的框被丢弃，这是 YOLO 的标准行为。

不引入可变长 buffer，是因为它会污染整个内存模型、graph 捕获以及与宿主语言的边界。用一个计数器解决，成本几乎为零。

### 5.2 预处理

letterbox 需要一个运行时才确定的缩放比例，取决于输入分辨率。

解法是把缩放比例与 padding 量作为 `derived` 表达式传入：

```json
"derived": {"scale": "min(64000/W0, 64000/H0)",
            "pad_x": "(640 - W0*min(64000/W0,64000/H0)/100)/2"}
```

这样同一个 KernelUnit 覆盖所有输入分辨率，不需要为每个分辨率生成 kernel。这正是 05 文档第 2 节符号表达式的用途。

### 5.3 多尺度 concat 的布局

`upsample + concat` 沿通道拼接。上游是 NHWC 时拼接方向连续，是 NCHW 时则不连续。

解法是由 Kernel Harness 的布局传播决定，把整个 backbone 统一在 NHWC 上，只在厂商卷积的边界做转换。

这会产生额外的转换 node，因此布局传播的目标函数必须包含转换代价，否则会做出"每个卷积都转一次"的糟糕决策。

这个问题在 TensorRT 里也存在，它有类似的格式传播。区别是：我们的决策在 Kernel Harness 中可读、可 diff、可解释，而不是藏在引擎的某个 pass 里。

---

## 6 验收

### 6.1 baseline 的选择

baseline 必须是最强的库，不是 PyTorch eager。

14 文档第 1 节的教训是：KernelBench 原协议下报告 1.43 倍，换成 TF32 baseline 后只剩 0.88 倍。用一个弱 baseline 自证成功，属于自欺。

必须同时对比：

| baseline | 配置 | 为什么必须有 |
|---|---|---|
| TensorRT | fp16，整图融合，tactic 搜索全开 | 唯一有意义的对手 |
| ONNX Runtime CUDA | fp16，图优化全开 | 开源对照 |
| PyTorch eager | fp16 | 下限参考，不能只用它 |

### 6.2 指标

| 指标 | 门槛 | 说明 |
|---|---|---|
| mAP@50-95，COCO val 子集 500 张 | 相对 PyTorch fp32 下降不超过 0.3 | 精度 |
| 端到端延迟，pre 加 infer 加 post，p50 与 p99 | 见 6.3 | 性能 |
| 峰值显存 | 不超过 TensorRT 的 110% | 显存 |
| 引擎开销占比 | kernel 时间占端到端不低于 85% | 引擎是否失控 |
| 确定性 | 两次运行逐位一致 | 可信度 |
| 形状覆盖 | coverage 声明范围内全部通过 | 不偷覆盖 |

### 6.3 性能目标

```
第一档，必须达到：不慢于 ONNX Runtime CUDA fp16。
                  达不到说明机制有问题。

第二档，应该达到：TensorRT fp16 的 1.00 到 1.10 倍，即差距不超过 10%。

第三档，理想：持平或略快于 TensorRT，优势来自 pre 与 post 也在同一个 graph 里。
```

第三档之所以可能成立，是唯一可信的赢面论证：TensorRT 的 engine 通常只包含推理主体，pre 与 post 在 engine 之外，需要额外的 launch 与同步。我们的 pre、main、post 是三个 plan，但可以在同一个 CUDA Graph 中重放，因为 NMS 用固定容量表达，没有控制流。端到端一个 graph，省掉的是几次 pipeline 边界同步与若干次 launch，在 2 ms 量级上是 5% 到 10%。

这是一个窄但真实的赢面。如果它不成立，就必须承认"我们打平或略慢，价值在于可维护性、可审计性和可移植性"，而不是性能。

### 6.4 测量协议

```bash
# 固定环境
nvidia-smi -lgc 1410,1410          # 锁频
export CUDA_LAUNCH_BLOCKING=0

# 每项测 3 轮，每轮 200 次迭代，取 p50
for b in trt ort ours; do
  bench --impl $b --warmup 50 --iters 200 --batch 1 --res 640x640
done

# 同时报告 --no-graph 与 --graph 两组数字
```

必须报告 `--no-graph` 的对照，否则无法区分"graph 省下的时间"与"融合省下的时间"。

---

## 7 分阶段落地

| 阶段 | 做什么 | 谁写 kernel | 目的 |
|---|---|---|---|
| M1-a | pre、post、三个 conv、一个 C2f | 人工手写 | 打通链路，验证 ABI 与 plan 执行 |
| M1-b | 全部三十个 KernelUnit | 人工手写，可从 CUTLASS 与 Triton 样例改写 | 建立性能基线，验证不依赖 Agent 也能跑 |
| M2 | 用 Agent 逐个替换 g11 到 g18 | Agent | 核心实验：Agent 能否达到手写的 90% |
| M3 | 全量 Agent、调优、多 shape | Agent | 验证成本模型与稳定性 |
| M4 | sm86，例如 Orin 或 RTX | Agent | 验证跨硬件 |

M1-b 是整个计划里最重要的一步，也是最容易被跳过的一步。

理由是：先用"手写 kernel 加显式 plan"跑通端到端，再让 Agent 以 plan 为输入做替换。如果跳过 M1-b 直接上 Agent，就会同时变化两个未知数，即引擎对不对、Agent 好不好，任何失败都无法归因。

---

## 8 预期收益

以 YOLOv8n fp16 在 A100 上端到端约 2.0 ms 为例，这个数字必须实测，此处只用于说明量级。

| 来源 | 量级 | 可信度 |
|---|---|---|
| 减少 launch 与 kernel 间隙，从约 120 个 kernel 降到 33 个并开启 graph | 100 到 300 μs | 高 |
| 消除中间张量写回，来自 C2f 与 SPPF 融合 | 50 到 150 μs | 中 |
| pre 与 post 进 graph，消除 pipeline 边界同步 | 30 到 80 μs | 中 |
| kernel 本身比 cuDNN 快 | 接近 0，甚至为负 | 高，见 P16 |
| 更好的布局，减少转置 | 0 到 100 μs | 低 |

总预期是相对 ONNX Runtime CUDA 提升 10% 到 25%，相对 TensorRT 在 -5% 到 +8% 之间。

这个结论应当在项目开始前就写下来：我们不承诺比 TensorRT 快很多，承诺的是在 TensorRT 覆盖不好的场景下，把部署周期从月降到天。

---

## 9 常见问题

| 做法 | 后果 |
|---|---|
| 只和 PyTorch eager 比 | 自欺，见 14 文档第 1 节 |
| 跳过 M1-b 直接上 Agent | 两个未知数同时变化，无法归因 |
| 为 NMS 引入可变长 buffer | 污染内存模型与 graph |
| 每个分辨率生成一个 pre kernel | 包数量与维护成本爆炸 |
| 每个卷积都做一次布局转换 | 转换开销吃掉融合收益 |
| 只测 batch 1 | 生产里常常是 4 或 8 |
| 只测 p50 | p99 才是线上体验 |
| 把"和 TensorRT 打平"当成失败 | 结合可维护性看，打平即可接受 |
