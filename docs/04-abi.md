# 04 KernelUnit ABI 规范（v1.0 草案，目标 S0）

本章为设计草案，尚未对应发布头文件。冻结条件见 03 文档第 0 节。`runtime/plan/entry.h` 实现内部入口元数据及静态绑定检查；目前可调用的是独立的 [实验 CPU ABI](../runtime/abi/experimental.h)，使用 `lm_ku_*_exp_v1` 符号，不是下面的目标 S0 ABI。实验版仅同步执行、stream 为 NULL、固定 workspace、无 alias，并按各 slot/入口声明保证对齐，未采用本章的统一 256 字节对齐。构建、调用和完整限制见 [16 文档](16-implementation.md)。

这份头文件是引擎与外部 kernel 之间唯一的硬边界。

设计上的取舍是：接口只描述缓冲区，不描述 kernel 内部。一个 KernelUnit 内部可以用多少 warp、是否使用 TMA、是否手写 PTX、是否为多 kernel 流水，引擎都不知道，也不应该知道。

---

## 1 设计约束

| 编号 | 约束 | 理由 |
|---|---|---|
| I1 | ABI 只承认四种跨界物：缓冲区、标量参数、workspace、stream | 这组概念已经十几年没有变化 |
| I2 | 引擎不解释 kernel 内部行为 | 否则会被锁死在某一代硬件上 |
| I3 | 结构体只允许末尾追加字段，配 `struct_size` | 这样 ABI 可以演进而不破坏二进制兼容 |
| I4 | 参数块对引擎完全不透明 | 因此支持任意 DSL、任意调优参数、任意内部结构 |
| I5 | 一个 KernelUnit 对应一个 `.so` 和一个入口符号 | 加载粒度清晰，可独立版本化、审计与替换 |
| I6 | 错误必须区分"引擎的错"和"kernel 的错" | 这是错误归因的前提 |
| I7 | 宿主侧 launcher 由 KernelUnit 自己提供，引擎不生成 | 引擎不需要知道 grid 和 block 怎么算 |

I7 是最关键的一条。它把"grid 怎么算、要不要用 persistent kernel、内部要不要并行"这些每半年就会变化的知识，全部挡在了边界之内。

---

## 2 单位与命名

```
KernelUnit（KU）   ABI 的最小可替换单元。一个融合后的计算块、一个预处理、一个 NMS。
                   对应磁盘上的 kernels/<name>/ 目录，编译产物是一个 .so。
implementation     某个 KernelUnit 内部的具体实现（ref、fast、tuned 变体）。
                   对应 .so 内的实现以及一个 impl_id。
variant            plan 层面选择的 impl 组合，见 05 文档第 5 节。
```

符号命名（S0）：

| 符号 | 必需 | 签名 | 说明 |
|---|---|---|---|
| `lm_ku_entry_v1` | 是 | `int32_t (const LmKernelArgs*, void* stream)` | 唯一入口 |
| `lm_ku_caps_v1` | 是 | `const LmKuCaps* (void)` | 能力声明，加载时必须调用 |

`.so` 内的其他符号一律 `static` 或使用 hidden visibility，避免符号冲突和 ABI 泄漏。

---

## 3 头文件

```c
/* loom/ku_abi.h  —  STABILITY: draft (target S0 after validation) */
#ifndef LOOM_KU_ABI_H
#define LOOM_KU_ABI_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LM_ABI_MAJOR 1
#define LM_ABI_MINOR 0
#define LM_ABI_VERSION ((LM_ABI_MAJOR << 16) | LM_ABI_MINOR)

/* ---------------- 数据类型 ---------------- */

typedef enum LmDType {              /* S0：只能追加 */
  LM_DT_INVALID = 0,
  LM_DT_F16     = 1,
  LM_DT_BF16    = 2,
  LM_DT_F32     = 3,
  LM_DT_F64     = 4,
  LM_DT_I8      = 5,
  LM_DT_U8      = 6,
  LM_DT_I32     = 7,
  LM_DT_I64     = 8,
  LM_DT_BOOL    = 9,
  LM_DT_F8E4M3  = 10,
  LM_DT_F8E5M2  = 11,
  LM_DT_I4      = 12,   /* 打包存储，见 4.4 */
  LM_DT_U4      = 13,
  /* 0x100 以上保留给实验性 dtype */
} LmDType;

typedef enum LmMemSpace {           /* S0：只能追加 */
  LM_MEM_DEVICE   = 0,
  LM_MEM_PINNED   = 1,   /* 页锁定主机内存，可异步拷贝 */
  LM_MEM_HOST     = 2,   /* 普通主机内存，只能同步拷贝 */
  LM_MEM_CONSTANT = 3,   /* 只读常量与权重，生命周期等于包的生命周期 */
} LmMemSpace;

typedef enum LmBackend {            /* S0：只能追加 */
  LM_BACKEND_CPU   = 1,
  LM_BACKEND_CUDA  = 2,
  LM_BACKEND_HIP   = 3,
  LM_BACKEND_METAL = 4,
} LmBackend;

/* ---------------- 缓冲区描述 ---------------- */

#define LM_MAX_DIMS 8

typedef struct LmBuffer {
  uint32_t  struct_size;   /* = sizeof(LmBuffer)，用于 ABI 演进 */
  uint32_t  dtype;         /* LmDType */
  void*     data;          /* 已含偏移的地址，引擎保证对齐 */
  uint8_t   rank;          /* 0..8，0 表示标量 */
  uint8_t   mem_space;     /* LmMemSpace */
  uint16_t  flags;         /* 位标志，见下 */
  uint32_t  layout_id;     /* 不透明布局标签，0 表示默认标签，非通配符，见 4.6 */
  uint32_t  _rsvd0;
  int64_t   dims[LM_MAX_DIMS];     /* 逻辑维度（元素数） */
  int64_t   strides[LM_MAX_DIMS];  /* 元素为单位；0 表示该维广播 */
  int64_t   elem_offset;           /* 相对 data 的元素偏移，通常为 0 */
} LmBuffer;

/* LmBuffer.flags */
#define LM_BUF_READ       0x01u   /* kernel 会读 */
#define LM_BUF_WRITE      0x02u   /* kernel 会写 */
#define LM_BUF_ALIASED    0x04u   /* 与其他 slot 物理重叠，引擎已校验 */
#define LM_BUF_INIT       0x08u   /* 内容已初始化，否则读它属于未定义行为 */

/* ---------------- 符号派生量 ---------------- */

/* plan 中每个 node 可以声明 derived 表达式，引擎在形状绑定后求值，
   以 int64 数组传入，避免 kernel 侧重复做整数运算。 */
typedef struct LmDerived {
  uint32_t        struct_size;
  uint32_t        n;
  const char* const* names;   /* n 个名字，仅用于调试 */
  const int64_t*  values;     /* n 个已求值结果 */
} LmDerived;

/* ---------------- 调用参数 ---------------- */

#define LM_CF_DEBUG          0x01u  /* 调试模式：允许 kernel 做额外断言或打印 */
#define LM_CF_STRICT_DET     0x02u  /* 要求确定性实现，非确定性 kernel 应返回错误 */
#define LM_CF_VALIDATE       0x04u  /* 校验模式：应选用 ref 实现，见 09 文档 */

typedef struct LmKernelArgs {
  uint32_t          struct_size;    /* = sizeof(LmKernelArgs) */
  uint32_t          n_in;
  uint32_t          n_out;
  uint32_t          _rsvd0;
  const LmBuffer*   in;             /* n_in 个，顺序等于 manifest.io 中 in 的顺序 */
  LmBuffer*         out;            /* n_out 个，顺序等于 manifest.io 中 out 的顺序 */
  void*             ws;             /* workspace 基址，至少 256 字节对齐；无需求时为 NULL */
  int64_t           ws_bytes;       /* workspace 可用字节数 */
  const void*       params;         /* 不透明参数块，布局由 manifest.params 定义 */
  int64_t           params_bytes;
  const LmDerived*  derived;        /* 可以为 NULL */
  uint64_t          call_flags;     /* LM_CF_* */
  void*             _rsvd1[4];      /* 保留，置 0 */
} LmKernelArgs;

/* ---------------- 能力声明 ---------------- */

#define LM_KUC_DETERMINISTIC    0x01u  /* 相同输入必得逐位相同输出 */
#define LM_KUC_SUPPORTS_INPLACE 0x02u  /* 允许 out 与某个 in 物理同址 */
#define LM_KUC_NEEDS_WS_QUERY   0x04u  /* 需要调用 workspace_bytes 才能得到大小 */
#define LM_KUC_SIDE_EFFECTS     0x08u  /* 有副作用，不可被消除 */

typedef struct LmKuCaps {
  uint32_t      struct_size;
  uint32_t      abi_version;      /* 必须与 LM_ABI_VERSION 的 major 兼容 */
  const char*   name;             /* KU 名，与 manifest 一致 */
  const char*   impl_id;          /* 当前 .so 提供的实现 id */
  uint32_t      backend;          /* LmBackend */
  uint32_t      arch_family;      /* 如 90；0 表示与架构无关 */
  uint32_t      arch_feature;     /* 如 90a 中的 a；0 表示无特殊要求 */
  uint32_t      flags;            /* LM_KUC_* */
  uint16_t      min_align;        /* 所有 buffer 的最小对齐要求（字节） */
  uint16_t      dialect;          /* 1=cuda_cpp 2=triton 3=tilelang 4=cute 5=ptx 0=prebuilt */
  const char*   tol_json;         /* 建议容差，如 {"rtol":1e-3,"atol":1e-5}；可以为 NULL */
  int64_t     (*workspace_bytes)(const LmKernelArgs*);  /* 可以为 NULL，见第 9 节 */
} LmKuCaps;

/* ---------------- 入口 ---------------- */

/* 返回 0 表示成功；1..255 表示 kernel 自定义错误，引擎映射为 LM_E_KERNEL；
   负值保留给引擎，kernel 不应返回负值。 */
typedef int32_t (*LmKuEntryFn)(const LmKernelArgs* args, void* stream);
typedef const LmKuCaps* (*LmKuCapsFn)(void);

int32_t           lm_ku_entry_v1(const LmKernelArgs* args, void* stream);
const LmKuCaps*   lm_ku_caps_v1(void);

#ifdef __cplusplus
}
#endif
#endif /* LOOM_KU_ABI_H */
```

---

## 4 字段说明

### 4.1 `LmBuffer`

| 字段 | 引擎保证 | kernel 应当遵守 |
|---|---|---|
| `struct_size` | 等于引擎编译时的 `sizeof(LmBuffer)` | 用 `struct_size` 判断可读字段范围，不要假设固定值 |
| `data` | 非 NULL，且地址按 `min_align` 对齐（取 256 与 caps 中的较大者） | 不要假设更大对齐 |
| `dtype` | 与 manifest 声明一致 | 遇到不认识的 dtype 返回非 0，不要猜测 |
| `rank` / `dims` | 实际运行时维度，符号已绑定 | 不要假设等于编译期维度 |
| `strides` | 元素为单位，0 表示该维广播 | 必须支持 stride 为 0，这是合法输入 |
| `elem_offset` | 相对 `data` 的元素偏移 | 计算地址时必须加上 |
| `layout_id` | 不透明整数，引擎只搬运不解释 | 若只支持某种 layout，在 manifest 声明 `required_layout`，不匹配时返回错误 |
| `mem_space` | 与实际分配一致 | 不要假设 device 内存可读 |

非连续张量是常态而不是例外。融合 kernel 的输入经常是上游输出的切片或转置视图。任何假设连续的实现，必须在 manifest 中声明 `requires_contiguous: true`，由引擎在调用前校验，不匹配则拒绝，要求外部生成显式转换节点。

### 4.2 `LmKernelArgs`

`in` 与 `out` 的顺序必须严格匹配 manifest 中 `io.fast.in` 与 `io.fast.out` 的声明顺序。顺序错了不会报错，只会算错。因此 plan 合成器必须从同一份 manifest 生成 plan 中的 slot 顺序，这也是要求 plan 由确定性代码生成的又一个原因。

`ws` 在 `ws_bytes` 为 0 时可以是 NULL，kernel 不得解引用。`params` 是只读的，引擎可能跨调用复用同一块内存。`_rsvd*` 必须被忽略。

### 4.3 `LmDerived`

用途是把 kernel 需要的标量从 kernel 内部的整数运算，搬到 plan 中的声明式表达式。

```c
// 一个 flatten 后的 GEMM 需要 M = N*H*W/4
// plan: "derived": {"M": "N*H*W/4", "K": "C"}
// kernel: int64_t M = args->derived->values[0];
```

没有直接传符号表，是因为符号名属于 plan 的私有概念，让 kernel 依赖它会把两层耦合起来。`derived` 按位置传值，更简单也更快。

### 4.4 子字节 dtype

`dims` 和 `strides` 始终以元素为单位。子字节类型的物理打包方式由 `layout_id` 决定，并必须在 manifest 的 `packing` 字段中声明，例如 `int4_row_major_lsb`。第一版不强制支持，但 ABI 预留了位置。

### 4.5 枚举与 dtype 约定

`LmDType`、`LmMemSpace`、`LmBackend` 都只能追加新值，既有值不得改变语义。kernel 遇到不认识的枚举值应当返回错误而不是猜测，否则一次扩展就会变成一次静默错误。

### 4.6 `layout_id` 的用法

`layout_id` 不是给引擎用的，是给 kernel 用的：

```
0        默认布局标签，非通配符；存储寻址仍由 strides 描述
1        NCHW
2        NHWC
3        blocked-32
>= 1024  自定义，必须在 manifest 中给出 layout 描述
```

引擎比较入口要求与 slot 的 layout_id，并原样传给 kernel，不解释 NHWC 等标签的数学含义。当前原型要求标签精确相同，包括 0；若未来增加任意布局约束，需显式表达，不能复用 0 暗示。

布局知识属于 kernel 和 plan，属于生产者，不属于引擎。

---

## 5 能力协商与加载协议

引擎加载一个 `.so` 的步骤如下，任何一步失败都拒绝加载并给出结构化错误：

```
1. dlopen(path, RTLD_NOW | RTLD_LOCAL)
2. caps = dlsym("lm_ku_caps_v1")                缺失 → LM_E_SYMBOL
3. entry = dlsym("lm_ku_entry_v1")              缺失 → LM_E_SYMBOL
4. caps->struct_size 覆盖已知前缀                否则 → LM_E_ABI
5. major(caps->abi_version) == LM_ABI_MAJOR      否则 → LM_E_ABI
6. caps->backend == 当前后端                     否则 → LM_E_UNSUPPORTED_HW
7. arch 兼容判定，见 03 文档 5.1 节              否则 → LM_E_UNSUPPORTED_HW
8. caps->min_align <= 引擎可提供的对齐           否则 → LM_E_ALIGN
9. caps->name 与 manifest 一致                   否则 → LM_E_MANIFEST
10. 若声明 LM_KUC_NEEDS_WS_QUERY，用假 args 调一次 workspace_bytes
    （此时必须允许 args->ws 为 NULL；返回负数则拒绝）
11. 把 entry 绑定到 (ku_name, impl_id) 索引
```

第 1 步使用 `RTLD_NOW` 而不是 `RTLD_LAZY`。宁可加载时失败，也不要等到第二个 batch 才崩。

### 5.1 一个实现一个 `.so`

另一种做法是一个 KernelUnit 一个 `.so`，内含多个实现，通过 `impl_id` 切换。第一版选择一个实现一个 `.so`，理由如下：

- 编译失败只影响一个变体，不阻塞其他变体。
- 可以针对不同 arch 使用不同编译选项，例如 sm_90a 与 sm_86。
- 可以只携带部分变体。
- 加载失败的影响面最小。

代价是包体积变大、`dlopen` 次数变多。实测 `dlopen` 约 0.1 到 1 ms，远小于推理时延，且可在加载期一次性完成。

---

## 6 错误模型

### 6.1 返回值约定

```
  0        成功
  1..255   kernel 自定义错误码，引擎映射为 LM_E_KERNEL，并原样透传具体值
  < 0      仅引擎使用，kernel 不得返回
```

不让 kernel 返回负值，是为了让引擎能无歧义地区分"kernel 主动报告的错误"和"引擎层的错误"。没有这个区分，错误归因就无法精确。

### 6.2 错误码

| 码 | 名称 | 含义 | 典型原因 |
|---|---|---|---|
| -1 | `LM_E_ABI` | ABI 不兼容 | 旧 `.so` 配新引擎 |
| -2 | `LM_E_SYMBOL` | 符号缺失 | 忘记 `extern "C"`，或符号被 strip |
| -3 | `LM_E_MANIFEST` | manifest 与实际不符 | 参数个数对不上 |
| -4 | `LM_E_SHAPE` | 形状或符号越界 | 请求的 shape 不在 plan 声明的域内 |
| -5 | `LM_E_ALIGN` | 对齐不满足 | `min_align` 未声明或过大 |
| -6 | `LM_E_OOM` | 分配失败 | arena 或显存不足 |
| -7 | `LM_E_KERNEL` | kernel 返回非 0 | 具体值见 `kernel_code` |
| -8 | `LM_E_DEVICE` | 设备侧错误 | 非法访存，或 XID 告警 |
| -9 | `LM_E_TIMEOUT` | 看门狗超时 | 死锁或死循环 |
| -10 | `LM_E_CONFORMANCE` | 差分校验失败 | 数值不正确 |
| -11 | `LM_E_MEM_PLAN` | 内存计划非法 | 活跃期重叠，或越界 |
| -12 | `LM_E_ALIAS` | 别名或读写冲突 | WAW、RAW 违规 |
| -13 | `LM_E_UNSUPPORTED_HW` | 硬件不满足 | arch、dtype 或驱动不符 |
| -14 | `LM_E_FORMAT` | 包格式不识别 | major 不匹配 |
| -15 | `LM_E_SIGNATURE` | 签名或溯源校验失败 | 无签名，或来源不在白名单 |
| -16 | `LM_E_DEGRADED` | 已降级运行，可配置为错误 | fast 不可用，走了 ref |

### 6.3 错误上下文

每个错误都要携带结构化上下文，这是 01 文档 P7 的要求：

```json
{
  "code": -10,
  "name": "LM_E_CONFORMANCE",
  "node_id": 17,
  "ku": "c2f_block",
  "impl": "sm90_f16_128x64_v3",
  "slot": 24,
  "slot_role": "out",
  "shape": [1, 71, 118, 42],
  "index": [0, 71, 118, 42],
  "expected": 0.5123, "actual": 0.5435,
  "abs_err": 0.0312, "tol_atol": 0.002,
  "hint": "该索引落在输入 slot 5 的 padding 区域，检查 fused conv 的边界处理"
}
```

`hint` 由规则引擎生成，不是由 LLM 生成。规则集见 09 文档第 7 节。

---

## 7 stream 的传递

`void* stream` 是不透明句柄，实际类型由后端决定：

| 后端 | 实际类型 |
|---|---|
| CUDA | `cudaStream_t` |
| HIP | `hipStream_t` |
| CPU | 引擎内部的 `LmCpuStream`，含线程池与任务队列 |
| Metal | `MTLCommandQueue*` |

CPU 后端的 kernel 可以通过该句柄提交并行任务，也可以直接串行执行。

约束：kernel 不得在 `stream` 上创建或销毁 stream，也不得在其中做同步等待。如果确实需要阻塞，在 manifest 声明 `may_block: true`，引擎会把它放到独立线程执行。引擎依赖"调用即返回"这一性质来维持调度。

---

## 8 参数打包

`params` 的字节布局由 manifest 声明，由 Kernel Harness 打包，引擎原样传递。

```json
"params": {
  "layout": "packed",
  "align": 8,
  "fields": [
    {"name": "stride",   "type": "i32", "default": 1},
    {"name": "pad",      "type": "i32", "default": 0},
    {"name": "dilation", "type": "i32", "default": 1},
    {"name": "groups",   "type": "i32", "default": 1},
    {"name": "act",      "type": "u8",  "default": 1},
    {"name": "tile_m",   "type": "u16", "default": 128},
    {"name": "tile_n",   "type": "u16", "default": 128},
    {"name": "split_k",  "type": "u16", "default": 1}
  ]
}
```

打包规则（S1）：

1. 按声明顺序排列。
2. 每个字段按 `min(自然对齐, align)` 对齐。
3. 总大小向上取整到 `align` 的倍数。
4. 出现未知字段时打包失败，不做静默丢弃。

kernel 侧读取：

```c
typedef struct { int32_t stride, pad, dilation, groups; uint8_t act;
                 uint16_t tile_m, tile_n, split_k; } Params;
static_assert(sizeof(Params) == 24);
const Params* p = (const Params*)args->params;
```

参数之所以是不透明字节而不是引擎理解的键值：引擎一旦理解参数含义（例如知道 `tile_m` 是什么），就会被引诱去做优化，例如自动调 tile，从而突破 01 文档 P8 的约束。不透明字节从机制上杜绝了这种诱惑。

---

## 9 workspace 约定

workspace 是 kernel 需要的临时显存，由引擎从共享 scratch arena 分配。manifest 用三种方式之一声明：

```json
"workspace": {"kind": "const", "bytes": 4096}
```
```json
"workspace": {"kind": "expr", "bytes": "ceildiv(N,64)*128*4"}
```
```json
"workspace": {"kind": "query"}
```

规则：

- `query` 模式必须在 `args->ws` 为 NULL 时也能返回正确大小，引擎在加载期会用假参数调用一次。
- 返回值只能依赖 args 中的描述信息，不得依赖 GPU 状态。
- 引擎保证 `ws` 按 256 字节对齐，且大小不小于声明值。
- workspace 不保证初始化为 0。需要零初始化的 kernel 必须自行 memset，或用 `LM_BUF_INIT` 检查。
- 同一 stream 上串行执行的相邻 node 之间，workspace 可能被复用，这是设计行为。

---

## 10 对齐与别名

### 10.1 对齐

引擎默认保证所有 buffer 的 `data` 按 256 字节对齐。若 kernel 需要更大对齐（例如某些指令要求 1024 字节），在 `caps.min_align` 中声明。

调试模式下引擎会做对齐扰动测试：把 arena 基址偏移 4、8、16 字节后重跑。任何依赖"恰好 256 字节对齐"的实现会在这里暴露。这是 Agent 生成代码最常见的错误之一，需要由基础设施兜住。

### 10.2 别名

```json
"io": {"fast": {
  "in":  [{"slot": "x", "rw": "r"}, {"slot": "w", "rw": "r"}],
  "out": [{"slot": "y", "rw": "w"}]
}}
```

- 声明 `rw: r` 的输入，引擎保证调用期间不被写。
- 声明 `rw: w` 的输出，引擎不保证其初始内容。
- 若希望支持原地计算（`y` 与 `x` 同址），必须声明 `LM_KUC_SUPPORTS_INPLACE`，并在 manifest 中标注 `inplace_ok: [0]`，表示 out[0] 可以与 in[0] 同址。
- 加载时引擎会校验：同一 node 内若某 slot 同时出现在 `in` 与 `out` 且未允许原地，返回 `LM_E_ALIAS`。

这是 Agent 生成的 plan 最容易出错的地方，尤其在 concat 与 split 之后。把这个校验放在引擎里是值得的。

---

## 11 确定性

| 状态 | 含义 | 来源 |
|---|---|---|
| 声明了 `LM_KUC_DETERMINISTIC` | 相同输入得到逐位相同的输出 | caps |
| 未声明 | 可能是非确定的，例如 atomic 归约或 split-K | 默认 |

引擎行为：

- `RunOptions.strict_determinism` 为真时，拒绝非确定性实现。
- 校验模式下总是跑两遍做逐位比对。

非确定性会让差分校验出现"有时过有时不过"的问题，是 Agent 闭环里最难排查的一类故障。宁可慢一点，也不要不稳定，除非应用明确接受。

---

## 11.1 数值契约与类型检查

buffer dtype 仅描述存储类型，不完整描述计算精度。manifest 应携带版本化的数值契约标识，记录必要的输入/输出 dtype、计算与累加类型、舍入/饱和规则、允许的近似，以及验证报告。融合单元内部存在多种精度时，允许使用分阶段描述，不能用一个全局 `compute_dtype` 掩盖差异；具体 schema 在原型验证后确定。

引擎校验 slot 与入口要求是否匹配、选中的实现是否满足要求的契约；它不能仅凭声明证明二进制遵守数值约定。核验由独立参考实现和离线测试承担。同一 KernelUnit 的可替换实现必须满足同一外部契约，不能把 FP16 buffer 当 FP32 传入，或通过隐式 Cast 实现降级。

量化契约还需表达 scale、zero point、轴/分组及参数绑定（如适用）；动态 scale 可由普通 KernelUnit 在运行期产生。Cast、Quantize、Dequantize 可作为独立 KernelUnit，也可在明确声明边界的融合单元内部执行。精度策略由 Kernel Harness 制定。

## 12 manifest（`ku.json`）

```json
{
  "name": "c2f_block",
  "version": "0.3.1",
  "desc": "YOLOv8 C2f block fused end-to-end (split -> convs -> concat)",
  "io": {
    "fast": {
      "in":  [ {"name": "x",  "dtype": "f16", "shape": ["N","H","W","C"], "layout": "nhwc"},
               {"name": "w0", "dtype": "f16", "shape": ["C","C1"],       "layout": "w"} ],
      "out": [ {"name": "y",  "dtype": "f16", "shape": ["N","H","W","C2"], "layout": "nhwc"} ]
    },
    "ref": { "in": ["同结构，可放宽 layout 与 stride 约束"], "out": ["同上"] }
  },
  "implementations": [
    {"id": "ref",                "kind": "ref",  "src": "ref/ref.cpp",    "min_align": 4},
    {"id": "sm90_f16_128x64_v3", "kind": "fast", "src": "src/c2f.cu",     "min_arch": 90,
     "dialect": "cuda_cpp", "params": {"tile_m": 128, "tile_n": 64}},
    {"id": "sm86_f16_64x64",     "kind": "fast", "src": "src/c2f_sm86.cu", "min_arch": 86,
     "dialect": "cuda_cpp"}
  ],
  "params": { "layout": "packed", "align": 8, "fields": [] },
  "workspace": {"kind": "expr", "bytes": "ceildiv(N*H*W, 64)*256*2"},
  "caps": {"deterministic": true, "min_align": 256,
           "tol": {"rtol": 1e-3, "atol": 1e-5},
           "requires_contiguous": false,
           "may_block": false},
  "derived": {"M": "N*H*W", "K": "C"},
  "meta": {"fused_from": ["split","conv","conv","concat"],
           "flops": "expr: N*H*W*C*C2*2",
           "provenance": "agent/kernel/2026-10-20/run_0417"}
}
```

字段说明：

| 字段 | 使用方 | 级别 |
|---|---|---|
| `io.fast` 与 `io.ref` 分别声明 | ref 可以更宽松，fast 可以更严格 | S1 |
| `implementations[].min_arch` | 指纹检查 | S1 |
| `implementations[].params` | 该实现的默认调优参数，可被 plan 的 variant 覆盖 | S1 |
| `caps.requires_contiguous` | 不匹配时拒绝，外部可生成显式连续化节点 | S1 |
| `meta.fused_from` | 仅供人工审计，引擎忽略 | S2 |
| `meta.provenance` | 溯源 | S1 |

`meta.fused_from` 被引擎忽略是有意为之。引擎知道"这个 kernel 融合了 split/conv/conv/concat"对执行没有帮助，但会诱导它做语义推理。所以放在 `meta` 下，并明确标注引擎不读。

---

## 13 两个完整示例

### 13.1 CUDA C++ 实现

```cpp
// src/relu.cu
#include "loom/ku_abi.h"
#include <cuda_fp16.h>

__global__ void relu_kernel(const half* __restrict__ x, half* __restrict__ y, int64_t n) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i < n) y[i] = __hmax(x[i], __float2half(0.f));
}

extern "C" int32_t lm_ku_entry_v1(const LmKernelArgs* a, void* stream) {
  const LmBuffer& x = a->in[0];
  LmBuffer&       y = a->out[0];
  int64_t n = x.dims[0];                     // 结构性要求：已 flatten
  if (x.strides[0] != 1) return 1;           // 错误码 1：不支持非连续
  int threads = 256;
  int blocks  = (int)((n + threads - 1) / threads);
  relu_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
      (const half*)x.data, (half*)y.data, n);
  return 0;
}

static const LmKuCaps kCaps = {
  sizeof(LmKuCaps), LM_ABI_VERSION, "relu", "default",
  LM_BACKEND_CUDA, 0, 0,
  LM_KUC_DETERMINISTIC, 256, 1, "{\"rtol\":0,\"atol\":0}", nullptr
};
extern "C" const LmKuCaps* lm_ku_caps_v1(void) { return &kCaps; }
```

编译：

```bash
nvcc -O3 -std=c++17 --shared -Xcompiler -fPIC -fvisibility=hidden \
     -gencode arch=compute_86,code=sm_86 -o librelu.so src/relu.cu
```

`-fvisibility=hidden` 配合 `extern "C"`，让只有两个符号对外可见。

### 13.2 Triton 实现

Triton 编译出的 `.so` 直接暴露的是 kernel 符号，需要一个薄 shim。shim 由 Kernel Harness 的模板生成，不由 Agent 手写。

```python
# src/relu_triton.py   —— Agent 只写这部分
import triton, triton.language as tl

@triton.jit
def relu_k(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    x = tl.load(x_ptr + off, mask=m)
    tl.store(y_ptr + off, tl.maximum(x, 0.0), mask=m)
```

```cpp
// src/shim.cpp —— 模板生成，Agent 不修改
#include "loom/ku_abi.h"

// Triton AOT 产出的 launcher，签名由模板按参数个数实例化
extern "C" void relu_k_launch(void* x, void* y, int64_t n,
                              int grid_x, int block_x, void* stream);

extern "C" int32_t lm_ku_entry_v1(const LmKernelArgs* a, void* stream) {
  const int64_t n     = a->in[0].dims[0];
  const int     BLOCK = 1024;
  const int     grid  = (int)((n + BLOCK - 1) / BLOCK);   // 调用时算 grid
  relu_k_launch((void*)a->in[0].data, (void*)a->out[0].data, n,
                grid, BLOCK, stream);
  return 0;
}

static const LmKuCaps kCaps = {
  sizeof(LmKuCaps), LM_ABI_VERSION, "relu", "triton_aot",
  LM_BACKEND_CUDA, 0, 0,
  LM_KUC_DETERMINISTIC, 256, 2, "{\"rtol\":0,\"atol\":0}", nullptr
};
extern "C" const LmKuCaps* lm_ku_caps_v1(void) { return &kCaps; }
```

`grid` 是在 `lm_ku_entry_v1` 里根据实际的 `n` 现算的，见第 1 节的 I7。这正是符号化 kernel 能覆盖所有 shape 的原因：引擎不需要知道 grid 是什么，同一个 `.so` 也能处理任意 `n`。

这套 launcher 包装应当固化成模板，Agent 只填 kernel body 部分。这是 01 文档 P5 在 kernel 层的落地。

---

## 14 编写时的注意事项

| 做法 | 问题 | 正确做法 |
|---|---|---|
| ABI 里出现 `block_size` 或 `num_warps` | 抽象层次过低，随硬件变化 | 放在 params 或 kernel 内部 |
| 引擎根据 dtype 推断容差 | 容差是应用级约定 | manifest 声明，应用配置可覆盖 |
| kernel 内部调用 `cudaDeviceSynchronize` | 破坏流水，可能死锁 | 只同步自己产生的内部 event |
| kernel 内部 new 或 delete 显存 | 破坏确定性，且拖慢 | 预分配或使用 workspace |
| 用 `getenv` 读调优参数 | 不可复现，线上失控 | 参数走 params |
| 导出包装过 `cudaMalloc` 的接口 | 引擎失去内存统计 | 只用引擎传入的 buffer |
| `lm_ku_caps` 返回的 `impl_id` 与实际不符 | 错误归因失效 | 从构建脚本注入 |
| 假设 rank 固定为 4 | 动态 shape 会崩 | 按 `rank` 分支 |
| 依赖 `data` 的绝对地址对齐 | 对齐扰动测试会崩 | 只用 `min_align` |
| 用 `float` 做索引运算 | 大 tensor 会溢出 | 一律用 `int64_t` |
