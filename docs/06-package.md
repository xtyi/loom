# 06 包格式、指纹与注册表（v1 草案）

当前已实现单独编号的 `loom-package-exp/1` 目录包（静态 CPU 子集），未实现本文的完整 v1 包、
指纹、签名或注册表。实际文件结构与加载命令见 [17 JSON 与目录包](17-json-package.md)。

包同时扮演四个角色：引擎的输入、Agent 的产出、部署的单元、兼容性的单位。

设计目标是让一个包像 ELF 一样：自描述、可校验、可签名、可互换、可长期归档。

---

## 1 设计约束

| 约束 | 原因 |
|---|---|
| 自包含 | 部署环境可能没有网络、没有 Python、没有 ONNX |
| 可审计 | kernel 源码必须保留，不能只留 cubin |
| 可验证 | 携带数值契约、参考来源、比较规则及验证数据，支持离线复验 |
| 可寻址 | registry 的 key 必须稳定且可计算 |
| 不可变 | 一个版本一旦发布就不允许修改，改动必须是新版本 |
| 可拒载 | 指纹不匹配时能干净地拒绝，而不是运行到一半崩溃 |

---

## 2 容器布局

```
model-yolov8n-h100.loom/            # 目录形式，用于开发期；发布时打包成 .loom 文件
├── package.json                   # 包元数据
├── fingerprint.json               # 特化声明，引擎读取
├── signature.sig                  # 签名
├── provenance.json                # 溯源
├── layouts.json                   # layout 名字到 id 的映射，仅 Kernel Harness 使用
│
├── plans/
│   ├── main.json                  # 主推理 plan
│   ├── prepack.json               # 加载时执行一次的权重预处理 plan
│   ├── pre.json                   # 预处理
│   ├── post.json                  # 后处理
│   └── mem/
│       ├── n1_640.json            # explicit 模式的内存计划
│       └── n4_640.json
│
├── kernels/
│   ├── conv_bn_silu/
│   │   ├── ku.json                # manifest
│   │   ├── params.json            # 打包后的默认参数
│   │   ├── src/
│   │   │   ├── conv_bn_silu.cu
│   │   │   └── ref.cpp
│   │   ├── build.json             # 构建配方
│   │   └── lib/
│   │       ├── sm90/libconv_bn_silu_sm90_f16_128x128.so
│   │       └── sm86/libconv_bn_silu_sm86_f16_64x64.so
│   ├── c2f_block/
│   └── ...
│
├── weights/
│   ├── w.bin                      # 全部权重，扁平拼接
│   └── w.idx.json                 # 每个权重的 name、shape、dtype、offset
│
├── goldens/
│   ├── case_n1_640/
│   │   ├── inputs/0.bin 1.bin
│   │   ├── outputs/0.bin
│   │   └── nodes/17.bin 23.bin ...    # 逐 node 快照，可选
│   └── case_n4_640/
│
└── bench/
    └── report.json                # 生成时的性能报告，仅供参考
```

源码必须保留。硬件换代后可以拿旧源码重编译，不必重新生成。这是应对"每代重写一遍"这类质疑的主要手段。

数值验证产物需关联 plan/实现/权重的内容哈希、PrecisionDecision、各 KernelUnit 的数值契约、参考执行器版本、参考数据的存储 dtype、比较规则及端到端质量报告。具体文件 schema 在原型期验证后确定。改变精度方案、转换边界或 kernel 实现后，旧报告不能自动沿用；静态契约匹配不等于二进制通过验证。生产默认不加载 golden，离线或按需诊断时使用。参考数据不统一要求 FP64，见 09 文档第 5.2 节。

---

## 3 `package.json`

```json
{
  "name": "yolov8n",
  "version": "1.3.0",
  "format": "loom/1.0",
  "engine_min": "0.4.0",
  "label": "YOLOv8n 640x640 fp16, H100/A100, batch 1-8",
  "requires": {
    "abi": ">=1.0 <2.0",
    "backend": "cuda",
    "min_arch": 86,
    "feature_sets": ["portable"],
    "driver": ">=535.54",
    "dtypes": ["f16", "f32"],
    "smem_min": 65536
  },
  "plans": {
    "main":    "plans/main.json",
    "prepack": "plans/prepack.json",
    "pre":     "plans/pre.json",
    "post":    "plans/post.json"
  },
  "io": {
    "pre":  {"in": ["uint8 image"]},
    "main": {"in": ["f16 NCHW"], "out": ["f16 3x3 8400"]},
    "post": {"in": ["f16 3x3 8400"], "out": ["boxes", "scores", "classes", "count"]}
  },
  "coverage": {
    "batch": [1, 2, 4, 8],
    "resolution": [[640, 640], [640, 480], [320, 320]],
    "policy": "fallback_ref"
  },
  "goldens": {"case_n1_640": "goldens/case_n1_640", "case_n4_640": "goldens/case_n4_640"},
  "profiles": {
    "prod":  {"signature": "required", "instrumentation": "off", "strict_determinism": true},
    "dev":   {"signature": "optional", "instrumentation": "node"}
  }
}
```

### `coverage` 的作用

它声明这个包验证过哪些形状。引擎在 run 时先查：

- 命中时正常执行，标记为 validated。
- 未命中但 `policy` 不是 `reject` 时执行，标记为 unvalidated，并发出 `LM_E_DEGRADED`，可配置为错误。
- `policy` 为 `reject` 时直接拒绝，并给出最接近的已知形状以及缺失的生成任务。

这条机制把"性能不可预测"变成"可声明、可检查"。生产环境可以配成 `reject`，测试环境配成 `fallback_ref`。

### 三个运行档位

| 档位 | 签名 | instrumentation | 确定性 | 用途 |
|---|---|---|---|---|
| `prod` | 必须 | off | 强制 | 线上 |
| `dev` | 可选 | node | 建议 | 开发调试 |
| `ci` | 可选 | full | 强制 | 回归测试 |

---

## 4 指纹

### 4.1 定义

```
target_fingerprint = {
  backend, arch_family, arch_feature_set,
  driver_major,
  dtypes_supported,
  smem_max, regs_max, sm_count_band,
  abi_major, plan_major
}
```

`sm_count_band` 使用区间（例如 `>= 80`）而不是精确数，因为 MIG 和切片会改变可见的 SM 数量。

### 4.2 判定

```
strict   所有字段都必须满足 requires
lenient  允许 driver 和 smem 不满足，但结果中必须标记 unvalidated
```

arch 的判定见 03 文档 5.1 节：必须用 `(family, feature_set)` 二元组，禁止按整数比较。`sm_90a` 的 kernel 不能跑在 `sm_90` 上，反之亦然。

### 4.3 为什么指纹必须进包

在本架构下，正确性依赖离线的若干假设：

- 某个 fast 实现可能使用了 `sm_90a` 的指令族。
- 某个内存计划可能假设了 228 KB 的 shared memory。
- 某个 variant 可能假设了 `sm_count == 132` 来做 tile 划分。

这些假设无法在运行时被安全推断，必须声明并检查。否则表现是"在 A100 上跑出了非法指令"，而且只在特定 shape 下崩。

---

## 5 溯源与签名

### 5.1 `provenance.json`

```json
{
  "generated_by": {
    "tool": "loom-harness/0.4.1",
    "agent": "claude-sonnet-4.5",
    "prompt_id": "kernel-c2f-v7",
    "prompt_hash": "sha256:...",
    "seed": 42,
    "started": "2026-10-20T09:12:00Z",
    "attempts": 14,
    "wall_clock_s": 4200
  },
  "reviewed_by": ["alice"],
  "source_model": {"format": "onnx", "hash": "sha256:..."},
  "toolchain": {"cuda": "12.4", "nvcc": "12.4.131", "triton": "3.1.0"},
  "licenses": [{"path": "kernels/c2f_block/src/c2f.cu", "scan": "clean", "spdx": "Apache-2.0"}],
  "artifacts": {"kernels/c2f_block/lib/sm90/libc2f.so": "sha256:..."}
}
```

强制要求（01 文档 P12）：

- `prod` 档位下，缺少 `provenance.json` 或缺少签名时拒绝加载。
- `licenses.scan` 不是 `clean` 的包在打包阶段失败，防止从 CUTLASS 等来源记忆出的代码带来许可问题。
- 所有 `.so` 的 sha256 必须记录，加载时校验，防止包被篡改。

### 5.2 签名

```json
{
  "algo": "ed25519",
  "signer": "loom-registry.internal",
  "signed_at": "2026-10-21T02:00:00Z",
  "manifest_hash": "sha256:...",
  "sig": "base64:..."
}
```

`manifest_hash` 覆盖 `package.json`、`fingerprint.json` 以及所有 `.so`。引擎维护签名者白名单，`prod` 档位下非白名单签名一律拒绝。

这不是过度设计。Agent 生成的 kernel 是任意代码，一旦允许在生产推理机上 `dlopen` 未经签名的 `.so`，整个供应链模型就不成立了。

---

## 6 权重与 prepack

### 6.1 权重与 plan 分离

权重体积大（YOLOv8n 约 3.2M 参数，fp16 约 6.5 MB），而 plan 需要能 diff 和审计，不能混在一起。同一份权重也可能对应多个硬件特化的包。

### 6.2 prepack 是一个 plan

这是"一切用 plan 表达"的第一个应用：

```
加载流程：
  1. 读 weights/w.bin 到显存，一次 H2D
  2. 执行 prepack plan，把权重从训练格式转成 kernel 需要的格式
     （layout 重排、量化、pad 到对齐边界）
  3. prepack 的输出 slot 标记为 persistent，生命周期等于引擎会话
```

这样做的好处是引擎里不需要出现任何格式转换代码，Agent 可以为每层选择它想要的权重布局，prepack 的耗时也可以单独测量，而它通常是启动时间的大头。

### 6.3 prepack 的时机

prepack 失败（例如权重 shape 与 plan 不符）必须在加载期报错，而不是等到第一次推理。`lm_package_load` 的语义就是：加载成功意味着所有非运行时依赖都已就绪。

例外情况是需要知道 batch 才能 prepack，这种设计应当避免，把 batch 放进符号即可。

---

## 7 第三方库作为 KernelUnit

ABI 不排斥厂商库。cuDNN、cuBLAS、CUTLASS 实例化产物都可以包装成 KernelUnit：

```json
// kernels/conv_implicit_gemm/ku.json
{
  "name": "conv_implicit_gemm",
  "implementations": [
    {"id": "cudnn_v9", "kind": "vendor", "vendor": "cudnn",
     "min_arch": 80, "src": "src/cudnn_wrap.cpp",
     "params": {"algo": 0, "math_type": 1}}
  ]
}
```

```cpp
// src/cudnn_wrap.cpp，由模板生成
extern "C" int32_t lm_ku_entry_v1(const LmKernelArgs* a, void* stream) {
  // 1. 用 a->in 和 a->out 构造 cudnnTensorDescriptor，stride 描述非连续布局
  // 2. 从 a->params 取 algo
  // 3. cudnnConvolutionForward(handle, ..., (cudaStream_t)stream)
}
```

这带来一个立场上的调整：本项目的目标不是用 Agent 取代厂商库，而是让 Agent 决定哪里该用库、哪里该自己写。

对于孤立的一层 3×3 s1 conv，cuDNN 的调优积累是压倒性的，Agent 应当直接选择它。对于 C2f 这类融合块，没有任何库提供现成 kernel，Agent 必须自己写。

所以 Agent 的核心价值是决定融合边界，而不是手写每一个卷积。这个定位降低了项目风险，也提高了第一版跑赢基线的概率。

配套要求：

- `kind: vendor` 的实现必须声明支持的 shape 范围，因为库的某些 algo 有严格限制。
- workspace 查询需要透传，cuDNN 的 workspace 与 algo 相关，通过 `workspace_bytes` 回调处理。
- 库的 handle 是 per-device 的，由 kernel 内部惰性创建并做引用计数，不能放进 `LmKernelArgs`，因为 ABI 需要稳定。

---

## 8 注册表

### 8.1 key 的设计

```
registry_key = hash(
  model_semantic_hash,      语义图的指纹，不含权重值
  package_version,          包自身版本
  target_fingerprint        目标硬件
)
```

不把 agent 和 prompt 放进 key，否则每次更换模型都会重复存储；这些信息放在 provenance 中用于追溯。

### 8.2 内容

| 存储 | 内容 |
|---|---|
| blob store | `.loom` 文件，不可变，按内容寻址 |
| index | `(model_hash, fingerprint) → [package_version...]` |
| metadata | 性能报告、校验状态、生成成本、评审记录 |
| 生命周期 | `draft → validated → promoted → deprecated` |

进入 `validated` 需要依次通过：

```
1. schema 校验
2. 全 coverage 的差分校验，含 padding、边界与扰动用例
3. 端到端任务指标达标，例如 mAP 下降不超过 0.2
4. 确定性检查
5. 许可扫描
6. 无跨 arch 的非法依赖
7. 人工评审 fast 实现
```

---

## 9 加载流程

```
lm_package_load(path, options):
 1. 解包与校验容器（magic、version、checksum）
 2. 检查 format.major                              → LM_E_FORMAT
 3. 校验 package.json schema                       → LM_E_MANIFEST
 4. 校验签名（按档位要求）                          → LM_E_SIGNATURE
 5. 校验 provenance（按档位要求）                   → LM_E_SIGNATURE
 6. 计算本机 fingerprint，比对 requires             → LM_E_UNSUPPORTED_HW
 7. 解析所有 plan 并做静态校验：
      每个 node 的 ku 与 impl 存在
      每个 slot 都被引用
      outputs 中的 slot 存在
      符号表完整
      每个 node 的 in/out 数量与 manifest.io 一致   → LM_E_MANIFEST
 8. 按需 dlopen 用到的 .so
 9. 校验 .so 的 sha256                              → LM_E_SIGNATURE
10. 执行 prepack plan
11. 返回不可变的 package 句柄，session 从它创建
```

第 7 步是"接口约定要完整"的体现：尽可能多的错误在加载期暴露，而不是在第一次推理，或者在运行数万次之后的某个 shape 上。

---

## 10 版本与不可变性

- 一个 `(name, version)` 一旦发布进 registry 就永不覆盖，改动必须新版本。
- 语义化版本的约定：
  - major：plan 格式不兼容，或输出语义变化
  - minor：新增 shape 覆盖、性能改进、新增实现
  - patch：修复 bug，数值不变

强调不可变的原因是 Agent 生成过程有随机性。如果允许覆盖，就无法二分定位"昨天好好的今天坏了"。

---

## 11 常见错误

| 做法 | 后果 |
|---|---|
| 只存 `.so` 不存源码 | 硬件换代后无法重编译，只能重新生成，成本翻倍 |
| 指纹只写 `min_arch` 整数 | `sm_90a` 与 `sm_90` 混淆，静默产生非法指令 |
| 用绝对路径引用权重 | 不可移植 |
| plan 里内联权重 | plan 无法 diff，体积过大 |
| 签名只覆盖目录而不覆盖 `.so` | 二进制可被替换 |
| 允许覆盖已发布版本 | 回归问题无法二分 |
| prepack 放在第一次推理时 | 首帧延迟出现尖刺，失败时机太晚 |
| coverage 声明为"全部形状" | 等于没有声明，指纹失去意义 |
