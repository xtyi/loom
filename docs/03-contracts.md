# 03 契约与稳定性分级

本文回答一个问题：哪些接口是稳定的，哪些不是。

答案不能凭感觉，需要一张可以执行的表。表中每一行都要有等级、变更规则、破坏成本和 CI 检查方式。

---

## 1 判定方法

判断一个接口元素能否长期不变，用下面三个测试。

### 测试一：抽象层次

假设两年后硬件换代，这个结构需要改字段，还是只需要加字段。

只需要加字段的，可以进入 S0 或 S1。需要改字段的，只能放在 S2 或 S3。

反例：如果 `LmKernelArgs` 里有一个 `int warps_per_block` 字段，两年后一定需要改，因为 warp 的概念本身会变。所以它不在 ABI 里，而在 kernel 内部。

### 测试二：生产者无关

如果出现一种全新的 kernel 生产方式（新的 DSL、新的编译器、更强的 Agent），这个接口是否需要修改。

不需要修改，说明它对"谁来生产"保持无知，是个好接口。需要修改，说明它泄漏了生产者的假设。

反例：如果 manifest 里有一个 `triton_config` 字段，那么换用 TileLang 就必须改格式。正确做法是一个不透明的 `params` 字段，由 KernelUnit 自己解释。

### 测试三：语义无关

为了使用这个接口，引擎是否需要理解模型语义。

不需要，说明合格。需要，说明违反了 00 文档 0.5 节的边界，不能进入引擎，只能是 S3 或更低。

反例：plan 里出现 `"op": "conv"` 字段，引擎就需要据此判断输出形状，语义发生泄漏。

---

## 2 四个等级

| 级别 | 承诺 | 破坏的代价 | 审批要求 |
|---|---|---|---|
| S0 冻结 | 永不破坏。只能末尾追加字段（配 `struct_size`）、追加枚举值（配 default 分支）、新增符号。历史版本必须始终能加载 | 已发布的包失效，线上事故 | 决策记录加负责人签字 |
| S1 稳定 | 语义与保证不变。允许兼容性扩展，如新增可选字段、新增变体、新增校验。破坏性变更需要 major 版本、迁移工具，以及一个 minor 周期的双写过渡 | 需要提升格式 major，重新打包 | 决策记录 |
| S2 实验 | 可以随时修改，但要有变更记录，且只能通过显式开关启用 | 无 | 常规评审 |
| S3 内部 | 实现细节，不做任何承诺 | 无 | 无 |

### S0 的可执行定义

S0 不是口号，它对应一个常驻的 CI 任务：

```
abi-corpus-test:
  输入：packages/corpus/**   每个历史版本至少保留一个包
  动作：用当前引擎加载每个包，跑 ref 路径，比对 golden
  要求：100% 通过，不允许跳过
```

这条规则把"承诺不破坏兼容性"从道德约束变成构建失败。每次发布 minor 版本时，自动把当次示例包加入语料库，只增不减。

---

## 3 接口元素清单

任何新增的跨界元素都要先在这里登记。

### 3.1 引擎与包

| 元素 | 级别 | 位置 | 变更规则 |
|---|---|---|---|
| `.loom` 容器布局（目录结构、必需文件） | S1 | 06 文档 | 只能新增文件或目录，路径语义不变 |
| `package.json` 必需键 | S0 | 06 文档 | 只能新增可选键 |
| `package.json` 可选键 | S1 | 06 文档 | 同上 |
| `requires` 字段名与含义 | S0 | 06 文档第 4 节 | 字段名冻结，判定逻辑属于 S1 |
| 签名与溯源文件 | S1 | 06 文档 | 字段可以增加 |

### 3.2 plan

| 元素 | 级别 | 位置 | 变更规则 |
|---|---|---|---|
| plan 顶层结构（`symbols`/`slots`/`nodes`/`variants`/`mem`/`outputs`） | S1 | 05 文档 | 可以增加字段，既有字段语义不变 |
| slot 字段（`role`/`dtype`/`shape`/`stride`/`layout`/`rw`/`alias_of`） | S1 | 05 文档第 3 节 | 同上 |
| node 字段（`id`/`ku`/`impl`/`in`/`out`/`params`/`ws`/`stream`/`after`/`derived`） | S1 | 05 文档第 4 节 | 同上 |
| 符号表达式语言的语法与求值语义 | S0 | 05 文档第 2 节 | 语法子集冻结，只能新增算子名；整数语义冻结 |
| variant 谓词语言 | S0 | 05 文档第 5 节 | 同上 |
| 内存计划语义（offset 表达式、生命周期定义、重叠判定） | S1 | 05 文档第 6 节 | 判定规则变更会影响正确性，需要决策记录 |
| 二进制编码格式 | S2 | 05 文档第 8 节 | 第一版只用 JSON |

符号表达式之所以是 S0 而不是 S1：它出现在已经发布的包里。plan 中写着 `"shape": "N*H*W/4"`，如果引擎某天改了 `/` 的语义（整除还是浮点），旧包就会算错，而且是静默错误。这条最容易在评审时被忽略。

### 3.3 引擎与 KernelUnit

| 元素 | 级别 | 位置 | 变更规则 |
|---|---|---|---|
| 入口符号名（`lm_ku_entry_v1`、`lm_ku_caps_v1`） | S0 | 04 文档第 2 节 | 新版本用 `_v2` 后缀，不改 `_v1` 的语义 |
| `LmBuffer` 与字段语义 | S0 | 04 文档第 4.1 节 | 只能末尾追加并配 `struct_size` |
| `LmKernelArgs` 与字段语义 | S0 | 04 文档第 4.2 节 | 同上 |
| `LmDerived` | S0 | 04 文档第 4.3 节 | 同上 |
| `LmKuCaps` | S0 | 04 文档第 5 节 | 同上 |
| `LmStatus` 错误码（负值域） | S0 | 04 文档第 6 节 | 只能追加，既有值语义不变 |
| `LmDType` 枚举 | S0 | 04 文档第 4.4 节 | 只能追加 |
| `LmMemSpace` 枚举 | S0 | 04 文档第 4.5 节 | 只能追加 |
| stream 以 `void*` 不透明传递 | S0 | 04 文档第 7 节 | 实际类型由后端决定 |
| 参数打包格式 | S1 | 04 文档第 8 节 | 布局规则可以扩展对齐方式 |
| workspace 约定 | S1 | 04 文档第 9 节 | |
| 对齐与别名约定 | S1 | 04 文档第 10 节 | |
| 确定性声明 | S1 | 04 文档第 11 节 | |

ABI 停留在"缓冲区"这一层而不继续向下抽象，理由见 00 文档 0.4 节：缓冲区、参数、workspace、stream 这组概念已经十几年没有变化。

### 3.4 KernelUnit manifest

| 元素 | 级别 | 位置 | 变更规则 |
|---|---|---|---|
| `ku.json` 顶层键 | S1 | 04 文档第 12 节 | 只增不改 |
| `implementations[]` 结构 | S1 | 同上 | |
| `io[]` 声明 | S1 | 同上 | |
| `params` schema | S1 | 同上 | 支持的类型集合可以扩展 |
| `workspace` 声明 | S1 | 同上 | |
| `caps` 声明 | S1 | 同上 | |
| kernel 源码方言（cuda_cpp/triton/tilelang/cute/ptx/prebuilt） | S2 | 同上 | 第一版支持 cuda_cpp、triton 与 prebuilt |
| 构建配方 `build.json` | S2 | 06 文档第 3 节 | Kernel Harness 内部约定 |

### 3.5 引擎与应用

| 元素 | 级别 | 位置 |
|---|---|---|
| `lm_package_load`、`lm_session_create`、`lm_session_run`、`lm_session_destroy` | S0 | 02 文档第 6 节 |
| `lm_validate` | S0 | 同上 |
| `lm_profile_begin`、`lm_profile_end`、`lm_report_json` | S1 | 同上 |
| `LmRunOptions` 等选项结构 | S1 | 同上 |
| `LmDeviceCaps` | S0 | 同上 |
| C++ 封装与 Python 绑定 | S3 | 语义跟随 C ABI |
| 结构化错误 JSON 的 schema | S1 | 09 文档第 7 节 |
| 观测输出格式 | S2 | 09 文档第 9 节 |

### 3.6 Kernel Harness 与 Agent

| 元素 | 级别 | 理由 |
|---|---|---|
| Kernel Harness 的 Python API | S3 | 内部工具，随 Agent 能力快速演化 |
| `SemanticGraph` | S3 | 只存在于 Kernel Harness，不进入运行时 |
| Agent 产物 schema（`FusionDecision`、`TuningReport`） | S2 | 需要相对稳定以便复用 prompt，但不承诺 |
| prompt 模板 | S3 | 经常修改 |
| `ConformanceReport` | S1 | 它是 Agent 闭环的反馈格式，随意修改会让历史日志不可比较 |
| registry 的 HTTP API | S2 | 部署形态未定 |
| CLI 子命令 `run`、`validate`、`pack`、`inspect` | S1 | 会被脚本和 CI 调用 |
| 其他 CLI 子命令与参数 | S3 | |
| 环境变量 `LOOM_*` | S3 | |

---

## 4 兼容性矩阵

修改某一项时，需要同时提升什么版本。

| 改动 | 需要提升 | 旧包能否直接运行 | 需要做的事 |
|---|---|---|---|
| `LmBuffer` 末尾加字段 | ABI minor | 能 | 旧 kernel 用 `struct_size` 判断 |
| `LmBuffer` 改字段类型或顺序 | 不允许 | — | 只能新增 `LmBuffer_v2` |
| 增加 `LmDType` 枚举值 | ABI minor | 能 | kernel 侧必须有 default 分支 |
| 增加 `LmStatus` 错误码 | ABI minor | 能 | 错误码映射表同步 |
| plan 增加可选字段 | plan minor | 能 | 引擎忽略未知字段 |
| plan 增加必需字段 | plan major | 不能 | 需要迁移工具 |
| plan 改变执行语义 | plan major 加决策记录 | 不能 | 需要迁移 |
| 增加符号表达式算子 | 不允许 | — | 需要新语法时引入 `expr_v: 2` 字段，不复用旧字段 |
| manifest 增加可选键 | KU minor | 能 | 旧 Kernel Harness 产出的包仍可读 |
| 增加后端类型 | ABI minor | 能 | 追加 `LmBackend` 值并检查 caps |
| 修改容差默认值 | 契约 minor 加变更记录 | 能，但结果可能变 | 需重跑全量校验 |
| 修改指纹算法 | 包格式 major | 不能 | registry 的 key 全变，需重建索引 |
| 修改目录布局 | 包格式 major | 不能 | 双读一段时间 |
| 修改 CLI 必需参数 | CLI major | — | 调用脚本会失效 |

引擎必须忽略 plan 和 manifest 中的未知字段，这是向前兼容的唯一实现方式。Kernel Harness 在设计新字段时，要把它做成可选且有保守默认值。

---

## 5 版本协商

四个独立版本号，不要混用：

```
engine_version   引擎发布版本（语义化版本）
abi_version      KernelUnit ABI，{major, minor}
plan_version     plan 格式，{major, minor}
package_format   包容器，{major, minor}
```

包在 `package.json` 中声明它需要的版本：

```json
{
  "format": "loom/1.0",
  "requires": {
    "abi": ">=1.0 <2.0",
    "backend": "cuda", "min_arch": 86,
    "driver": ">=535",
    "dtypes": ["f16"]
  }
}
```

引擎按以下顺序判定，任何一步失败都直接拒绝，不做降级：

```
1. format.major 是否等于引擎支持的最大 major      否 → LM_E_FORMAT
2. abi 范围与引擎支持范围是否相交                  否 → LM_E_ABI
3. backend 是否匹配                                否 → LM_E_UNSUPPORTED_HW
4. min_arch 是否满足                               否 → LM_E_UNSUPPORTED_HW
5. driver 是否满足                                 否 → LM_E_UNSUPPORTED_HW
6. 所需 dtype 是否全部支持                         否 → LM_E_UNSUPPORTED_HW
```

### 5.1 arch 不能按整数大小比较

`sm_90` 与 `sm_90a` 互不兼容，因为 `a` 表示一组不可移植的架构特性；`sm_86` 与 `sm_89` 的二进制则互相兼容。

因此必须用 `(arch_family, feature_set)` 二元组比较：

```
arch_family:  90
feature_set:  { portable: 90, arch_specific: 90a }    两者正交

兼容条件：kernel.require_family <= device.family
          且 kernel.require_feature_set 是 device.feature_set 的子集
```

用整数大小比较是这类问题最常见的错误来源，而且失败方式往往是静默的非法指令。

---

## 6 弃用流程

S0 元素没有弃用流程，只能追加新版本符号（`_v2`）。S1 元素按下面的节奏处理：

```
minor N     标记 deprecated，更新文档、编译告警与变更记录，功能仍完整可用
minor N + 1 运行时打日志（每进程一次），Kernel Harness 提供自动迁移
minor N + 2 只允许在 major 版本中删除，且必须提供迁移工具与迁移文档
```

---

## 7 变更评审

修改公共接口的 PR 需要回答：

- 这个元素在本文第 3 节里吗。不在的话先补登记。
- 它的等级是哪个，依据是第 1 节里的哪个测试。
- 这次是加字段还是改字段。改字段的话，为什么不能新增 `_v2`。
- 旧包会不会失效，`abi-corpus-test` 跑过没有。
- 是否更新了 `CHANGELOG.md` 的 interface 小节。
- 如果是 S1 的语义变更，是否增加了新的校验规则和对应测试。

---

## 8 为什么值得投入

一个反面教材是 ONNX 的 opset 演进和 TensorRT 的 engine 版本绑定，两者都导致"升级工具链等于全部重做"。

在 Agent 生产 kernel 的模式下，这个风险被放大了。包的绝对数量会远多于传统引擎（每个模型、每个硬件、每类 shape 一个包），单个包的生命周期会很短（Agent 升级后重新生成），但基础设施必须比包活得久（CI、registry、监控、部署系统不会跟着重写）。

包的易变性越强，接口约定的稳定性就越值钱。这也是本文把 S0 和 S1 的边界划得比较保守的原因。
