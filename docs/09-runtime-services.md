# 09 运行时服务

02 文档讲的是模块边界。本文讲每个服务具体提供什么保证、怎么实现、开销多少。

这些服务与模型语义无关，也与硬件代际无关，是引擎里真正需要长期维护的部分。

---

## 1 服务与开销

| 服务 | 默认开销 | 提供的保证 | 失败时的错误码 |
|---|---|---|---|
| 内存（arena、liveness） | 每个 shape 一次性约 20 μs | 无重叠、对齐、无越界 | `LM_E_MEM_PLAN`、`LM_E_OOM` |
| 并发（stream、event、graph） | 每次 run 约 3 到 8 μs | 依赖正确、事件无泄漏 | `LM_E_ALIAS`、`LM_E_DEVICE` |
| 动态形状（绑定、variant、coverage） | 约 5 到 15 μs | 域外必报错，不静默 | `LM_E_SHAPE` |
| 校验 | 仅在显式调用时 | 定位到 node、slot、索引 | `LM_E_CONFORMANCE` |
| 归因 | off 约 0，node 档 1% 到 2% | 逐 node 耗时与瓶颈分类 | — |
| 降级 | 0 | 始终存在可执行路径，且可见 | `LM_E_DEGRADED` |
| 确定性与回放 | off 约 0 | 逐位可复现 | — |
| 超时检测 | 0 | 挂死能报出，但不能中断 | `LM_E_TIMEOUT` |

总开销目标：对 YOLOv8n（约 40 个 node，端到端 2 ms），引擎自身不超过 40 μs，即 2%。分解来看，绑定与 variant 选择在缓存命中时约 12 μs，依赖图约 6 μs，提交约 40 次乘以 0.2 μs，事件约 4 μs。开启 graph 后提交降到约 3 μs。

---

## 2 内存服务

### 2.1 arena 模型

```
LmSession
 └── InflightSlot[0 .. max_inflight)
      └── Arena（一块大的显存分配，大小由 plan 的 arena 表达式决定）
           ├── persistent 区：prepack 输出与权重，生命周期等于 session
           ├── input 与 output 区：零拷贝时可以直接使用调用方的指针
           └── temp 与 workspace 区：按 liveness 复用
```

一次性分配，永不 realloc。原因是 `cudaMalloc` 是同步的、昂贵的，且会造成显存碎片；而推理的显存需求在形状绑定后是完全已知的。这是本架构相对动态图框架的一个结构性优势。

### 2.2 liveness 分配

算法见 05 文档第 6.4 节。两条实现约束：

1. 分配结果按 `(variant_id, 符号值)` 缓存。同一 batch 反复推理时不能每次都跑分配。
2. 分配必须是纯函数，相同输入产出相同偏移表。这是回放与确定性检查的前提。

### 2.3 对齐与对齐扰动

引擎保证 `data` 按 256 字节对齐。Agent 生成的 kernel 常犯的一个错误是隐式依赖更大对齐，例如手写向量化读时假设 1024 字节对齐，或者在 `reinterpret_cast<float4*>` 前没有检查。

调试模式下做对齐扰动测试：

```
对 delta 取 0、4、8、16、32、64、128：
    arena_base 加上 delta
    run 并与 golden 比较
```

只在 `delta=0` 时正确的实现会在这里被抓住。这类测试 Agent 很难自己想到，需要由基础设施提供。

### 2.4 canary

调试模式下在每个 slot 两端各插入 512 字节的 canary 区并填充魔数，运行后校验：

```
[SENTINEL 512B][ slot data ][SENTINEL 512B]
```

这能抓住绝大多数越界写，即使它没有立刻导致崩溃。相比 `compute-sanitizer`（慢 10 到 50 倍），canary 只增加约 2% 开销和少量显存，适合作为每次 CI 的常规检查。

### 2.5 零拷贝

如果调用方提供的 `LmTensor.data` 已经是设备指针，且 dtype、layout、stride 与 plan 期望一致，引擎直接把它作为 slot 的存储，跳过拷贝。

这个优化在多路视频场景里很关键：解码器输出的 GPU frame 常常已经在显存里，多一次 H2D 就是多 0.5 ms。

条件是全部满足：`mem_space` 为 device、对齐不小于 256 字节、`strides` 与 slot 声明一致、`layout_id` 一致。任一不满足则插入拷贝并记一条警告，不静默降级。

---

## 3 并发服务

### 3.1 依赖图构建

```
输入：按 id 排序的 nodes，以及各 slot 的读写集合
输出：每个 node 的 wait 集与 signal 集

算法，复杂度 O(n·k)，k 为平均读槽数：
  last_writer[slot]     = -1
  readers_since_w[slot] = []
  按顺序遍历 node：
    wait = {}
    对 node.reads 中的 s：若 last_writer[s] >= 0，加入 wait
    对 node.writes 中的 s：若 last_writer[s] >= 0，加入 wait
                           wait 并入 readers_since_w[s]，然后清空
    wait 并入 node.after
    对 writes 中的 s：last_writer[s] = node.id
    对 reads 中的 s：readers_since_w[s].add(node.id)
```

合法性校验，任一条失败即报错：

- 某个 slot 在写入之前被读，报"读到未初始化数据"，指出 node 与 slot。
- 环检测，因为 `after` 边可能造环。
- 跨流的 RAW 如果因为缺 `after` 而乱序，自动插边，不报错，因为这是引擎能修的。

自动插边与报错的分界是：如果 Agent 漏了声明，但引擎能从数据流推出正确顺序，就修正并记录警告；如果数据流本身有歧义，例如同一 slot 被两个 node 写且没有顺序，报错。

### 3.2 事件池

`cudaEventCreate` 约 1 到 2 μs，每次 run 都创建会吃掉预算。做法是维护事件池：`acquire` 从空闲链表取，没有则创建；`release` 放回，不销毁；池上限 256，超出才销毁。

事件必须 record 之后才能 wait，且同一事件被重新 record 会覆盖之前的记录，因此池中的事件在被重录前必须已经无人等待。这个约束由"每个 in-flight 槽位独立池"来保证。

### 3.3 graph 捕获

```
触发：graph key 第二次出现，第一次只记录不捕获
流程：
  1. 在该槽位的主 stream 上开始捕获
  2. 逐 node 提交，与普通路径使用同一份代码
  3. 跨流依赖在捕获中变成图的内部节点
  4. 结束捕获，得到 graph 句柄，存入 LRU
重放：
  1. 在主 stream 上启动 graph
  2. 外部输出的同步通过 event 完成
```

graph 的硬约束：

| 约束 | 对应处理 |
|---|---|
| 捕获期间不能 `cudaMalloc` 或 `cudaFree` | arena 预分配，见 2.1 |
| 捕获期间不能调用同步接口 | kernel 约定禁止，见 04 文档第 7 节 |
| 捕获期间不能访问 pageable 主机内存 | 使用 pinned 内存 |
| 捕获期间不能创建或销毁 stream | 由引擎保证 |
| 图中的 kernel 参数被固化，包括指针 | arena 偏移必须稳定，同一 key 必须对应同一偏移 |

最后一条是最危险的隐式约束。如果同一 graph key 下 arena 偏移发生变化，重放会使用旧指针。因此 graph key 必须包含决定偏移的全部信息，且偏移分配必须是确定性的。

收益方面，YOLO 的 40 个 node 普通提交约 8 μs，加上每次 launch 约 3 到 5 μs 的 GPU 间隙；graph 重放后整个图只需一次 launch。在 2 ms 端到端里省 5% 到 8%，小 batch 场景更明显。

### 3.4 graph 与动态形状的边界

已知的事实是：形状变化不适合直接放进 graph，因为捕获会把 kernel 参数（含 grid 维度）按值固化。

本项目的处理方式：

| 情况 | 做法 |
|---|---|
| 同一 shape 反复推理，这是绝大多数生产场景 | 一个 graph，长期重放 |
| shape 有少量离散取值，例如 batch 取 1、2、4、8 | 每个 shape 一个 graph，LRU 16 |
| shape 连续变化，例如可变分辨率视频流 | 不开 graph，走普通提交，或按形状分桶 |
| 只有指针变化、结构不变 | 复用同一 graph 结构，用 `cudaGraphExecKernelNodeSetParams` 更新参数 |

需要明确的是：graph 是执行优化，不改变语义。开不开 graph 必须产出逐位相同的结果，这条断言应当在每晚 CI 中运行。

---

## 4 动态形状服务

### 4.1 符号绑定

符号从输入绑定，绑定关系在 plan 中显式声明：

```json
"symbols": [{"name": "N", "bind": {"slot": 0, "dim": 0}},
            {"name": "H", "bind": {"slot": 0, "dim": 1}}]
```

规则：

- 每个符号恰好绑定一次。
- 被绑定的维度必须与实际 buffer 的 `dims` 一致。若两个 slot 都声称 dim0 是 N，则两者必须相等，否则 `LM_E_SHAPE`。
- 禁止从间接信息推导符号，例如从输出指针反推。这会让错误延迟暴露。

### 4.2 两级检查

```
第一级：plan.domain 的谓词，例如 "H % 32 == 0"。回答"能不能正确执行"。
第二级：package.coverage 的枚举，例如 batch 属于 {1,2,4,8}。回答"验证过没有"。
```

两级分离很重要。域外表示必须拒绝，因为会算错；覆盖外表示可以执行但没验证过，需要标记。

### 4.3 域外错误的可行动性

`LM_E_SHAPE` 的错误信息必须回答"怎么办"：

```json
{
  "code": -4, "name": "LM_E_SHAPE",
  "failed_predicate": "H % 32 == 0",
  "bound": {"N": 1, "H": 642, "W": 640},
  "suggestion": "输入 H=642 不能被 32 整除。可以把输入 pad 到 672，
                 或生成一个覆盖 H%32!=0 的 variant",
  "nearest_valid": {"H": 640}
}
```

`nearest_valid` 由规则计算，取最近的可满足值，不是由 LLM 生成。

### 4.4 降级时的形状处理

如果某个形状只命中 `ref` 实现，没有 `fast`，引擎：

1. 执行 `ref`，结果正确但可能慢 5 到 20 倍。
2. 返回 `LmRunResult.degraded = 1`，并附带原因。
3. 发出指标，可被监控告警。

降级不能静默。一个悄悄慢了 15 倍的推理服务，比一个报错的服务更难排查。

---

## 5 校验服务

"跑一遍 diff"不是校验，校验是持续对抗，所以这一节做得比较重。

### 5.1 四级校验

| 级别 | 内容 | 成本 | 运行时机 |
|---|---|---|---|
| L0 静态 | schema、连接性、符号完整性、in/out 数量与 manifest 一致 | 加载期，小于 1 ms | 每次加载 |
| L1 差分 | 逐 node 对比 fp64 golden | 一次 ref 加一次 fast | 每次打包、每次 CI |
| L2 统计 | 形状扫描乘以输入分布扫描 | N 次 | 每晚 CI |
| L3 任务级 | 端到端指标，如 mAP、mIoU、top-1 | 数据集规模 | 发布门禁 |

### 5.2 误差预算标定

问题在于 fp16 下"等价"没有定义：容差松了放过 bug，紧了误杀，而 Agent 会精确卡边界。

做法是不猜容差，而是标定容差：

```
golden   = ref 实现在 fp64 下的输出，这是唯一真值
floor    = |fp16_ref - fp64_ref|          该 node 在 fp16 下的固有误差
tol_node = max(k * floor, dtype_epsilon)  k 默认 2.0，可配置
判定     = |fast - fp64_ref| <= tol_node
```

三个性质：

1. 真值是 fp64，不是 fp16。否则 golden 自己就是误差来源，Agent 可以"正确地"复现 golden 的错误。
2. 容差来自该 node 的固有误差，而不是全局常数。累加很长的归约天然误差大，不该被判错；纯 elementwise 的 kernel 误差大就是真 bug。
3. 容差在打包时算好写进包，Agent 不能修改。Agent 能改的是实现，不是判据。这从机制上堵死了卡容差这条路。

对于无法用 fp64 表达的情况，例如 int8 量化或纯整数运算，改用精确相等或相对 L2 误差小于 1e-3。

### 5.3 反作弊

Agent 会优化"通过校验"这个目标，而不是"正确"这个目标，所以校验必须包含它猜不到的维度。

| 手段 | 抓什么 | 做法 |
|---|---|---|
| 边界与填充探针 | 只在有效区域算对，padding 区乱算 | 在输入的 padding 区填极端值（±65504、NaN），检查输出不受影响 |
| 依赖探针 | 少算一部分，或跳过某些输入 | 逐元素扰动输入，输出必须有响应 |
| 对齐扰动 | 隐式对齐假设 | 见 2.3 |
| canary | 越界写 | 见 2.4 |
| 分布扫描 | 只在训练分布上正确 | 均匀、正态、极值、稀疏、全零、单调 |
| 形状扫描 | 只在一个分辨率上正确 | coverage 全域，包含奇数尺寸与 1×1 |
| 两次逐位比对 | 非确定性 | 验证 `deterministic` 声明 |
| 峰值显存核算 | 用 workspace 爆显存，或放弃原地计算 | 记录逐 node 显存水位，超过 ref 的 110% 判失败 |
| 语义无关性测试 | 引擎里藏了语义 | 见 00 文档 0.5 节 |

依赖探针值得单独说明，它的实现是：

```
对某个 node 的第 i 个输入 slot：
  随机选 8 个元素，各加 delta = eps * max(|x|, 1)
  重跑该 node，检查输出至少有一个元素的变化超过容差
  若 8 次扰动全部无变化，报"input i 未被使用"，疑似 bug
```

成本是每个 node 跑 9 次。YOLO 的 40 个 node 约 360 次 run，几分钟内可以完成，适合每晚 CI。

### 5.4 定位报告

这是反馈给 Agent 的主要内容：

```json
{
  "verdict": "fail",
  "first_failing_node": 17,
  "ku": "c2f_block", "impl": "sm90_f16_128x64_v3",
  "slot": 24, "role": "out",
  "tol": {"rtol": 1e-3, "atol": 1e-5, "calibrated_from": "fp16_ref_vs_fp64"},
  "max_abs_err": 0.0312,
  "max_rel_err": 0.0841,
  "worst_index": [0, 71, 118, 42],
  "error_histogram": {"p50": 3.1e-5, "p99": 1.2e-3, "max": 3.1e-2},
  "spatial_pattern": "错误集中在最后两行与最后两列，占 71% 的超差元素",
  "suspected": ["边界处理：输入未 padding，kernel 读了越界区域并 clamp",
                "第 2 个输入未被使用，依赖探针在该输入上无响应"],
  "repro": "loom-cli validate --pkg yolov8n.loom --case case_n1_640 --node 17"
}
```

`spatial_pattern` 和 `suspected` 由规则引擎产生，不是 LLM。它按错误元素的分布形态匹配特征：集中在边界、集中在某个通道、全图均匀、按 block 粒度分布。这四类形态对应四类典型 bug，对下一步动作的指导价值很高。

### 5.5 成本

| 对象 | 一次完整 L1 的成本 |
|---|---|
| YOLOv8n，40 个 node，单 shape | ref 一次约 15 ms，fast 一次约 2 ms，比对约 5 ms，快照 IO 约 80 MB |
| 完整 L2，24 个 shape 乘 6 种分布 | 约 144 次乘以 20 ms，约 3 s，加上 IO |

校验不是轻活，但可以自动化。它应当占到工程量的 30% 到 40%，这个预算要写进计划。

---

## 6 归因服务

没有可靠的归因，Agent 的优化就是随机游走。

### 6.1 逐 node 计时

| 档位 | 手段 | 开销 | 精度 |
|---|---|---|---|
| off | 无 | 0 | — |
| node | 每个 node 两端插 event，异步收集 | 1% 到 2% | ±1 μs |
| full | 加 canary、快照、带宽计数 | 10% 到 30% | 仅供调试 |

node 档与 graph 模式互斥，因为 graph 重放后无法逐 node 插 event。因此 profile 时会自动关闭 graph，这一行为必须报告给用户，否则"profile 时 2.1 ms、线上 2.0 ms"的差异会让人困惑。

### 6.2 不依赖 NCU 的瓶颈分类

manifest 可以声明 roofline 元数据，只是数字，不涉及语义：

```json
"meta": {"roofline": {"flops": "N*H*W*C*C2*2",
                      "bytes": "N*H*W*(C+C2)*2 + C*C2*2"}}
```

引擎在 profile 时用数值除以时间，得到该 node 的实际 FLOPS 与带宽，与设备峰值比较，直接分类：

| 分类 | 判据 | 对 Agent 的含义 |
|---|---|---|
| 访存受限 | 带宽利用率大于 70%，且 FLOPS 利用率小于 30% | 再优化计算没有意义，应考虑减少数据搬运或融合更多生产者 |
| 计算受限 | FLOPS 利用率大于 60% | 优化分块，或换更快的指令 |
| 延迟受限 | 两者都低，且耗时接近固定开销 | 网格太小、launch 开销或尾部效应 |
| 未知 | 没有声明 roofline | 提示 Agent 补充声明 |

这个分类不需要 NCU，只需要时间加上声明的数字。它让 Agent 在没有 profiler 权限的环境里也能判断方向。

### 6.3 规则诊断

| 现象 | 信号 | 推断 |
|---|---|---|
| launch 开销主导 | 平均 kernel 时长小于 20 μs，且 launch 次数超过 30 | 应当融合，或开启 graph |
| 尾部效应 | 最后三个 node 占总时间超过 35% | 网格太小，或串行尾部 |
| 流间不均衡 | 多流重叠时各流总时长差超过 40% | 划分不均，应调整 stream 分配 |
| 同步停顿 | 相邻 node 之间空隙超过 2 μs | event 过多，或跨流依赖太密 |
| 重复计算 | 同一 slot 被多次写，且只被读一次 | plan 中有冗余节点 |
| 拷贝瓶颈 | H2D 或 D2H 时间超过 15% | 零拷贝未生效，或输入布局不匹配 |

这些规则是确定的、可以单元测试的、可以解释的。

---

## 7 诊断与降级

### 7.1 降级顺序

```
1. tuned 实现，最优参数
2. 默认参数实现
3. 其他 arch 的实现，前提是 min_arch 满足且 feature set 兼容
4. ref 实现，正确但慢
5. 拒绝，返回 LM_E_SHAPE 或 LM_E_UNSUPPORTED_HW
```

每一级降级都要记录事件、置 `degraded=1`、带上原因。策略可配：

```json
{"degrade": {"policy": "warn", "max_level": 4}}
```

`policy` 可以是 `ignore`、`warn`、`error`，`max_level` 表示允许降到第几级。

### 7.2 设备上下文失效

CUDA 的一个事实是：一次非法访存之后，整个 context 不可再用，此时任何接口都返回同样的错误，重试没有意义。

引擎必须：

1. 检测到 `LM_E_DEVICE` 时，把整个设备标记为失效。
2. 之后所有 session 的 run 立即失败，快速返回，不占用资源。
3. 只有进程重启才能恢复。
4. 上报高优先级指标，并 dump 最近一次成功 run 的 plan 快照。

否则线上表现是"所有请求都超时"，而真实原因是三十分钟前的一次越界写。这是最难排查的一类事故。

### 7.3 超时检测

GPU 上无法杀死一个挂死的 kernel，这一点不能假装可以做到。能做到的是：

| 能力 | 做法 |
|---|---|
| 检测挂死 | 等待完成事件时带超时 |
| 上报 | 返回 `LM_E_TIMEOUT`，附当前 node |
| 隔离 | 标记设备失效，让新请求快速失败 |
| 事后定位 | 保存超时前最后若干 node 的完成时间戳，指向卡住的位置 |
| 中断 kernel | 做不到，除非用 MIG 加整卡重置，代价是所有 session 失效 |

因此超时检测的目标从"恢复"改成"快速失败加精确定位"。这个定位比一个假装能恢复的实现更有用。

---

## 8 确定性与回放

### 8.1 三档

| 档位 | 保证 | 实现 |
|---|---|---|
| best-effort | 不保证 | 默认 |
| strict | 逐位可复现 | 拒绝非确定性 KernelUnit，禁用使用 atomic 归约的实现 |
| check | 跑两遍逐位比对，不同则报错 | 用于 CI |

### 8.2 录制与回放

`--record=DIR` 把一次 run 的输入、符号值、选中的 variant、所有 buffer 的地址映射落盘；`--replay=DIR` 在无设备环境下用 CPU 后端跑 ref 实现并比对。

用途：

- 线上事故后拿到录制文件，在本地复现，不需要相同的 GPU。
- 与新版本引擎对比同一输入是否产出同一输出。
- 作为 ABI 语料测试的用例来源。

体积控制：默认只录输入与输出，YOLO 一次约 6 MB。中间 buffer 需要显式开启，约 80 MB。

---

## 9 观测导出

| 格式 | 用途 |
|---|---|
| Chrome trace | 人工看时间线，观察流重叠与空隙 |
| 结构化 JSON | 机器分析，CI 断言 |
| Prometheus 指标 | 线上监控 |

必须暴露的最小指标集：

```
lm_run_duration_us{quantile=50,99}
lm_run_degraded_total
lm_graph_hit_ratio
lm_arena_used_bytes / lm_arena_capacity_bytes
lm_kernel_time_fraction
lm_validate_failures_total{reason}
```

`lm_kernel_time_fraction` 是判断引擎开销是否失控的唯一指标。它小于 85% 时，说明引擎自己吃掉了太多时间，这是引擎的 bug，不是模型的问题。

---

## 10 常见问题

| 做法 | 后果 |
|---|---|
| 用 `cudaDeviceSynchronize` 做同步 | 破坏流水，无法定位 |
| 每次 run 重新 `cudaMalloc` | 显存碎片、同步、慢一个数量级 |
| 容差用全局常数 | 要么放过 bug，要么误杀 |
| golden 用 fp16 计算 | 真值本身是错的，校验失去意义 |
| 降级不报告 | 线上悄悄慢 15 倍 |
| graph 模式下做 profile | 数据全为 0，或只有一次 launch |
| 假装能杀死挂死的 kernel | 事故时白等 |
| 校验只用一个随机输入 | Agent 会学到分布外取巧 |
| 错误信息只有错误码 | Agent 无法行动，人也无法排查 |
