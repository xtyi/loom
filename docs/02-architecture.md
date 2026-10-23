# 02 软件架构设计

本文描述目标系统的模块划分、依赖方向和运行模型；当前已实现的子集见 16 文档。以下性能数字是待验证的设计估计，不是当前原型的测量结果。

架构的目标是让 00 文档 0.5 节的边界（引擎不依赖模型语义）在代码结构上可以被检查，而不是靠开发者自觉。

---

## 1 系统结构

系统分四层，从上往下看。

应用层是调用 `lm_session_run` 的服务代码，负责请求循环、批处理与对外接口。它只看得见 tensor、session 和错误码。

运行时 `loom-rt` 用 C++17 实现，不依赖 Python、编译器和算子库，内部有九个模块。`api` 提供唯一的 C ABI 并对参数做校验；`package` 负责解包、manifest 校验、签名与指纹比对；`plan` 负责解析 JSON、求值 `SymExpr`、选择 variant；`exec` 负责构建依赖图、插入事件、提交调用与 graph 捕获；`mem` 负责 arena 与 liveness 分配、别名校验；`loader` 负责 `dlopen` 与符号绑定；`backends` 抽象设备操作；`conform` 提供差分校验与定位；`observe` 提供逐 node 计时与归因。另有 `diag` 生成结构化错误，`util` 提供基础工具。

设备层提供显存、stream 和 event。运行时通过 `backends` 与它交互，上层看不到 CUDA 类型。

KernelUnit 层是编译好的 `.so`，每个暴露一个 `lm_ku_entry_v1` 入口。它内部可以调用 CUDA、Triton 或 TileLang 的编译产物，也可以只是对 cuDNN 的薄包装。运行时不知道里面是什么。

另有一条离线链路，通过公开接口复用运行时能力：Kernel Harness（`loom-harness`，Python）把模型转成 plan，驱动 Agent 产出 KernelUnit，跑校验和调优，最后打包成 `.loom`。Agent 是外部模型，只通过结构化产物与 Kernel Harness 交互。

依赖单向：Kernel Harness 可通过 `loom-cli` 或可选 Python 绑定调用公开 runtime API，不访问内部模块；runtime 不依赖 Harness、Python 或绑定库。两者通过包格式、KernelUnit ABI 和公开宿主 API 交互。部署时应用拿到的就是 `.loom`，运行时读取它、加载其中的 `.so`、按 plan 执行。

runtime 内没有模型编译器或算子生成器。离线工具链承担语义转换、精度规划、kernel 编译与打包。

---

## 2 三处需要分开看的东西

评审改动时，先判断它属于下面哪一类：

| 类别 | 内容 | 变化速度 | 归属 |
|---|---|---|---|
| 运行期逻辑 | 加载、内存、并发、执行、校验、诊断 | 慢，十年尺度 | 运行时（S0/S1） |
| 接口定义 | ABI、plan 格式、包格式、manifest schema | 慢 | 运行时与文档（S0/S1） |
| 生成期产物 | kernel 实现、融合决策、调优参数、golden | 快，随模型与硬件变化 | Kernel Harness 与 Agent（S2/S3） |

如果一项改动属于第三类，却需要修改第一类的代码，说明抽象没有划清，应当先修抽象。

---

## 3 模块划分与依赖方向

### 3.1 运行时模块

| 模块 | 职责 | 依赖 | 不应包含 |
|---|---|---|---|
| `api/` | C ABI 实现、参数校验、错误翻译 | 其它全部 | 业务逻辑 |
| `package/` | 解包、manifest 校验、签名与溯源、指纹比对 | `plan`、`util` | 执行 kernel |
| `plan/` | JSON 解析、`SymExpr` 求值、谓词求值、variant 选择、DAG/schedule 静态校验 | `util` | 访问设备 |
| `exec/` | 使用已校验依赖与 schedule，事件插入、提交、graph 捕获与重放 | `plan`、`mem`、`loader` | 解释算子含义 |
| `mem/` | arena、liveness 分配、别名校验、对齐、canary | `util` | 分配单个 tensor |
| `loader/` | `dlopen`、符号绑定、caps 协商、sha256 校验 | `util` | 生成 launcher |
| `backends/` | 设备操作抽象（分配、拷贝、stream、event、graph、caps） | `util` | 把 CUDA 类型泄漏到上层 |
| `conform/` | 差分校验、扰动测试、依赖探针、定位报告 | `exec`、`mem` | 修改执行语义 |
| `observe/` | 逐 node 计时、计数器、trace 导出、归因规则 | `exec` | 在默认模式下引入可观开销 |
| `diag/` | 错误码、结构化错误、诊断规则 | `util` | 输出非结构化日志 |
| `util/` | 容器、字符串、JSON、日志、哈希 | 无 | 依赖上层 |

依赖方向固定为：`api → {package, exec, observe, conform} → {plan, mem, loader} → backends → util`。禁止反向依赖，也禁止跨层调用，CI 用 include 图检查保证。

`mem/` 不负责分配单个 tensor，只管理 arena 内的偏移。这样所有显存都来自少数几次大块分配，便于统计和排查。

### 3.2 后端抽象

`backends/` 是唯一允许出现 CUDA 类型的地方。上层只看到抽象接口：

```cpp
class Device {
public:
  virtual Status alloc(size_t bytes, void** p) = 0;
  virtual Status free(void* p) = 0;
  virtual Status memcpy_h2d(void* dst, const void* src, size_t n, StreamRef s) = 0;
  virtual Status memcpy_d2h(void* dst, const void* src, size_t n, StreamRef s) = 0;
  virtual Status synchronize(StreamRef s) = 0;

  virtual StreamRef stream_create(int priority) = 0;
  virtual EventRef  event_create(bool timing) = 0;
  virtual void      event_record(EventRef, StreamRef) = 0;
  virtual void      stream_wait(StreamRef, EventRef) = 0;

  virtual Status graph_begin(StreamRef) = 0;
  virtual Status graph_end(StreamRef, GraphRef*) = 0;
  virtual Status graph_launch(GraphRef, StreamRef) = 0;

  virtual const DeviceCaps& caps() const = 0;
  virtual void* native_stream(StreamRef) const = 0;   // 传给 kernel 的 void*
};
```

`native_stream()` 返回的就是 ABI 中那个不透明的 `void* stream`。`cudaStream_t` 不会出现在 `backends/` 之外。

### 3.3 Kernel Harness 模块

| 模块 | 职责 | 需要语义 | 可调用 LLM |
|---|---|---|---|
| `ingest/` | 第一版 ONNX 转成 `SemanticGraph`，其他来源按需适配 | 是 | 否 |
| `precision/` | 精度方案、转换边界、量化参数和质量预算 | 是 | 否，首版使用显式策略 |
| `refrun/` | 原始模型基线及符合目标数值契约的独立参考 | 是 | 否 |
| `partition/` | 融合分组决策 | 是 | 可以，结果经结构校验 |
| `layout/` | 布局传播与重排决策 | 是 | 否 |
| `mem/` | 显式内存计划 | 是 | 否 |
| `emit/` | 生成 plan，确定性合成器 | 是 | 否，硬性禁止 |
| `gen/` | 驱动 Agent 编写 kernel | 是 | 是 |
| `check/` | 调用引擎的校验服务 | 否 | 否 |
| `tune/` | 变体与参数调优 | 是 | 可以 |
| `pack/` | 打包、签名、许可扫描 | 否 | 否 |
| `registry/` | 上传、下载、索引 | 否 | 否 |

`emit/` 不允许 import 任何 LLM 客户端库，CI 检查 import 图。原因是 plan 的连线属于机械变换，出错后不会报错、只会算错，必须由确定性代码生成（01 文档 P5）。

---

## 4 线程与并发模型

### 4.1 基本模型

一次 `lm_session_run` 分为三段：绑定与校验、提交、等待完成。三段分别发生在不同位置，开销也差一个数量级以上。

| 步骤 | 执行方式 |
|---|---|
| bind、domain、variant | 同步，纯 CPU，约 5 到 20 μs |
| mem 计算与校验 | 同步，形状相同则直接复用缓存 |
| deps 构建依赖图 | 同步，与 node 数量成正比 |
| graph 命中时重放 | 异步提交 |
| graph 未命中时逐 node 提交 | 异步提交，launch 立即返回 |
| 返回 `LmRunHandle` | 此刻 GPU 可能仍在执行 |
| `lm_run_wait` | 阻塞在该槽位的完成事件上 |

`run` 默认不阻塞，应用可以在 GPU 执行时准备下一批数据。需要同步语义时传 `sync=1`。

### 4.2 in-flight 槽位

```cpp
struct InflightSlot {
  Arena             arena;       // 独立显存分区
  vector<StreamRef> streams;     // 与 plan 中出现的 stream id 一一对应
  vector<EventRef>  events;
  GraphCache        graphs;      // 每个槽位独立，避免跨线程竞争
  Status            state;       // free / running / done
};
```

`LmSessionOptions.max_inflight` 决定槽位数量，默认 1。每个槽位持有独立的 arena，大小按 plan 的 arena 表达式乘以槽位数确定。

不共享 arena 有三个原因：共享需要加锁或原子操作，会进入关键路径；共享会产生显存碎片；CUDA Graph 要求捕获时的指针在重放时依然有效，动态分配器无法保证这一点。

代价是显存占用随 `max_inflight` 线性增长。YOLO 单次 arena 约 80 MB，四个槽位是 320 MB，通常可以接受。

### 4.3 线程安全约定

| 对象 | 约定 |
|---|---|
| `LmPackage` | 只读，多线程安全 |
| `LmSession` | 非线程安全，一个 session 对应一个线程 |
| `LmRunHandle` | `wait` 可以被不同线程调用一次 |
| `lm_device_caps` | 线程安全 |

需要并发时创建多个 session，共享同一个 package，权重只加载一次。这条约束让引擎内部不需要加锁。

---

## 5 一次 run 的执行流程

一次 `lm_session_run` 分成八步：前五步为主机侧准备，第六步提交设备工作，第七步返回异步句柄，第八步由调用方等待完成。

1. 绑定符号。`api` 从输入 tensor 的 `dims` 推出 `N`、`H`、`W` 等符号值，绑定关系由 plan 的 `symbols` 声明。
2. 域检查。`plan` 校验 `domain` 中的谓词，不满足则返回 `LM_E_SHAPE` 并指出失败的谓词。
3. 选择 variant。`plan` 按当前符号值和硬件条件选出 impl 集合与内存计划。
4. 构建依赖图并确定 schedule。合并数据边与 `after`，检查无环；使用确定性拓扑顺序或验证外部提交顺序。
5. 计算或校验内存。依据 schedule 和完成关系计算/验证偏移，再准备必要的同步。缓存身份必须包含图、实现、schedule、内存计划及形状。
6. 提交。graph 命中时执行重放；否则按已验证的 schedule 逐个调用 KernelUnit。
7. 返回句柄。此时 GPU 可能仍在执行，`lm_session_run` 默认不阻塞。
8. 等待与返回。`lm_run_wait` 阻塞在该槽位的完成事件上，返回耗时与是否降级。

准备、提交和设备执行应分别测量；依赖构建与内存规划的缓存收益、单次提交开销需要在后端实现后验证。

---

## 6 对外接口

以下为 C ABI 设计草案，尚未实现；未来落在 `include/loom/rt.h`。当前原型接口是内部 C++ API，不承诺二进制兼容。

```c
typedef struct LmPackage LmPackage;
typedef struct LmSession LmSession;
typedef struct LmRun     LmRun;

typedef struct LmTensor {
  void*    data;
  uint32_t dtype;
  int32_t  rank;
  uint8_t  mem_space;
  int64_t  dims[8];
  int64_t  strides[8];     /* 0 表示紧凑布局 */
  int64_t  elem_offset;
} LmTensor;

typedef struct LmLoadOptions {
  uint32_t struct_size;
  const char* profile;              /* "prod" | "dev" | "ci" */
  int      allow_unvalidated;       /* 指纹不匹配时是否允许加载 */
  const char* signer_whitelist_path;
} LmLoadOptions;

typedef struct LmSessionOptions {
  uint32_t struct_size;
  int32_t  device_id;
  int32_t  max_inflight;
  uint32_t flags;
} LmSessionOptions;

typedef struct LmRunOptions {
  uint32_t struct_size;
  int32_t  sync;
  int32_t  strict_determinism;
} LmRunOptions;

typedef struct LmRunResult {
  uint32_t struct_size;
  int64_t  wall_us;
  int32_t  degraded;
  int32_t  variant_id;
} LmRunResult;

/* 生命周期 */
LmPackage* lm_package_load(const char* path, const LmLoadOptions*, LmError**);
void       lm_package_release(LmPackage*);

LmSession* lm_session_create(LmPackage*, const LmSessionOptions*, LmError**);
void       lm_session_destroy(LmSession*);

/* 执行 */
int lm_session_run(LmSession*,
                   const LmTensor* in,  int32_t n_in,
                   LmTensor*       out, int32_t n_out,
                   const LmRunOptions*,
                   LmRun** handle, LmRunResult*);

int lm_session_run_plan(LmSession*, const char* plan_name,
                        const LmTensor*, int32_t, LmTensor*, int32_t,
                        const LmRunOptions*, LmRun**, LmRunResult*);

int  lm_run_wait(LmRun*, int64_t timeout_us);
void lm_run_release(LmRun*);

/* 校验与观测 */
int  lm_validate(LmSession*, const LmValidateOptions*, LmReport**);
void lm_report_free(LmReport*);
const char* lm_report_json(LmReport*);

int lm_profile_begin(LmSession*, const LmProfileOptions*);
int lm_profile_end(LmSession*, LmProfileResult*);
int lm_profile_export(LmSession*, const char* path, const char* format);

/* 设备与诊断 */
const LmDeviceCaps* lm_device_caps(int32_t device_id);
const char* lm_error_message(LmError*);
const char* lm_error_json(LmError*);
int         lm_error_code(LmError*);
void        lm_error_free(LmError*);
```

错误用 `LmError**` 返回，而不是只返回错误码。原因是 01 文档 P7 要求错误携带 node、slot、索引、期望值和实际值，这些内容必须放在堆对象里。所有函数仍然返回整数错误码，`0` 表示成功，简单调用点可以只看返回值。

---

## 7 关键实现选择

D1 惰性加载 `.so`。 一个包可能包含多种 arch、上百个 `.so`，但一个会话只会用到其中一份。加载期只解析 manifest，首次执行时按 variant 需要的 impl 批量 `dlopen`。代价是首次执行有约 20 ms 的加载尖刺，因此提供 `lm_package_warmup(pkg, sample_inputs)` 供应用在启动阶段预热。预热必须显式调用，不能隐式预加载，否则冷启动时间不可控。

D2 Graph 缓存的 key。 key 由四部分组成：`variant_id`、绑定的符号值、impl 集合的哈希、stream 分配的哈希。符号值必须进入 key，因为 kernel 的 grid 可能由形状决定。这意味着动态 shape 下缓存按形状累积，需要 LRU，默认保留 16 个。

D3 事件插入策略。 只对跨 stream 的依赖插入 event，同 stream 内的顺序由提交顺序保证。event 对象池化复用，避免每次 `cudaEventCreate`。

D4 liveness 分配结果要缓存。 在依赖/schedule 校验后计算偏移表；key 包含图/实现、schedule（含 stream 分配）、内存计划和形状，不能仅用 variant id 与 shape。开销需要实测。

D5 校验模块编译进运行时，但不在正常路径上执行。 它只在调用 `lm_validate` 时激活，会增加约 80 KB 体积，好处是线上也能做现场诊断。`prod` 档位下不加载 golden 数据，以节省内存。

---

## 8 仓库结构

```
loom/
├── docs/                        # 本文档集
├── include/loom/
│   ├── ku_abi.h                 # S0：KernelUnit ABI
│   └── rt.h                     # S0：宿主 API
├── runtime/
│   ├── api/  package/  plan/  exec/  mem/  loader/
│   ├── backends/{cuda,hip,cpu}/
│   ├── conform/  observe/  diag/  util/
│   └── tests/
│       ├── unit/                # 纯 CPU，无 GPU 也能跑
│       ├── integ/               # 需要 GPU
│       └── corpus/              # ABI 兼容性测试包，只增不减
├── cli/                         # loom-cli：run/validate/pack/inspect/bench
├── harness/                     # Kernel Harness（Python）
│   ├── loom_harness/{ingest,precision,refrun,partition,layout,mem,emit,gen,check,tune,pack,registry}/
│   ├── prompts/                 # Agent prompt，S3
│   └── templates/               # kernel 脚手架与 shim 模板
├── examples/yolov8n/            # 端到端示例
├── benchmarks/
└── tools/
```

`runtime/tests/unit/` 必须能在没有 GPU 的机器上运行，用 CPU 后端加假设备。GPU runner 成本高、可获取性差，这条约定能把 CI 成本降低一个数量级。

---

## 9 决策记录

关键选择各自的理由记录在 `docs/decisions/` 下。每份对应一个在实现前就应当确定的决定，避免日后反复讨论。

| 编号 | 决定 | 要点 |
|---|---|---|
| 0001 | 不做图编译与算子生成 | 范围裁剪的依据 |
| 0002 | KernelUnit 为契约原子，一个实现一个 `.so` | 失败隔离与 arch 分离 |
| 0003 | C ABI 加 `dlopen`，不做源码级 JIT | 运行时不含编译器 |
| 0004 | 执行 DAG 与 schedule 分离，引擎不改写模型语义 | 依赖正确性与确定性 |
| 0005 | 结构由程序生成，kernel 由 Agent 生成 | 限制 LLM 的作用范围 |
| 0006 | 每个 in-flight 槽位独立 arena | 无锁、偏移确定 |
| 0007 | 厂商库作为一等 KernelUnit | 不在厂商最强的领域硬拼 |
| 0008 | grid 与 block 由 KernelUnit 的 launcher 计算 | 符号化 kernel 的前提 |
| 0009 | 精度决策离线，runtime 执行数值契约 | 正确性与模型质量分别验证 |
| 0010 | C++17 核心、C ABI、可选 Python 绑定 | 语言与依赖边界 |

---

## 10 与既有引擎的区别

| 引擎 | 中心 | 本项目 |
|---|---|---|
| TVM | IR 与 schedule 原语 | 接口约定（IR 在 Kernel Harness 内，且不追求通用） |
| TensorRT | engine 文件与 tactic 选择 | 接口约定（tactic 在离线阶段决定） |
| ONNX Runtime | execution provider 生态 | 接口约定（provider 退化为 KernelUnit） |
| IREE | 可执行格式与 HAL | 接口约定（HAL 保留，编译外置） |
| PyTorch | eager 执行加编译器栈 | 接口约定（不试图统一两者） |

这些系统的共同点是在"把图变成程序"上投入巨大，本项目不做这件事。

差异在于假设：它们假设程序由编译器产生，本项目假设程序由 Agent 产生。两种假设需要的基础设施恰好是同一套接口约定，区别只在由谁生产。
