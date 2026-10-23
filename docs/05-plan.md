# 05 Plan 格式规范（v1 草案，目标 S1）

Plan 是可执行程序的描述：节点已绑定到编译好的 KernelUnit，边表达数据依赖与额外的完成顺序约束。它不包含待运行时选择实现的 Conv 等模型算子。由离线工具完成语义转换后，确定性地生成 plan；这也是离线编译流程的一部分。

## 1 Plan 的角色与输入边界

| 层次 | 内容 | 消费者 |
|---|---|---|
| 模型语义图 | 算子、属性、逻辑张量与模型权重，如 ONNX | Kernel Harness |
| 执行 DAG | 已绑定实现的调用节点、slot、参数和依赖 | runtime |
| schedule | 可选的具体提交顺序与 stream 分配 | runtime，校验后执行 |

加载时提供执行图、KernelUnit 二进制、manifest、权重及目标设备要求；调用时提供实际输入、shape/标量与输出绑定。图级 `inputs` 列出调用方每次提供的 slot，顺序即调用接口顺序；每项必须存在、不重复且为 external。权重/常量可为 external 但不列入 inputs，由加载阶段单独提供。`outputs` 保留最终输出顺序。已编译二进制仍须由 adapter/launcher 满足 Loom ABI、内存和异步执行契约。

DAG 是依赖约束，schedule 是具体执行方案，两者不能混为一谈。缺少 schedule 时，使用稳定的 Kahn 拓扑排序：每次从就绪集合取最小 node id，全部使用 stream 0。显式 schedule 的格式见第 4.2 节。第一版仅支持单 stream，不启用自动多流启发式。

ONNX、StableHLO 适合作为离线输入，不是任意预编译 KernelUnit 的通用调用 ABI。Loom 自定义小型执行格式，前端通过适配器接入；参考来源见 14 文档第 7 节。

### 首批实现子集

首批 C++ 原型使用不可变值的执行 DAG：每个非外部 slot 恰有一个生产者，外部 slot 由调用方或权重加载提供且只读。暂不实现 alias、原地更新、persistent 状态、多流、控制流和动态输出；下文这些字段属于后续设计，不能据此声称原型已支持。当前内存中的 Graph 也不是 JSON schema 的完整实现。`prepare` 仅校验拓扑，`prepare_program` 要求完整的 tensor 与 EntryContract；实验 CPU Session 在此基础上加载动态库并同步执行，每次调用独立分配中间 buffer，具体限制见 16 文档。

### 一切计算用 KernelUnit 表达

| 场景 | 表达方式 |
|---|---|
| 固定权重转换 | 优先离线完成；需要设备端预打包时用初始化 plan |
| 预处理、后处理、主推理 | 普通执行 plan |
| Cast、量化/反量化、动态 scale 计算 | 显式节点或声明了数值契约的融合实现 |
| 精度策略选择、校准、误差预算 | Kernel Harness 离线决策，不由 runtime 自动推断 |

---

## 2 符号表达式语言 `SymExpr`

形状到运行时才知道，所以 plan 中的维度、偏移和大小都是符号表达式。

### 2.1 语法

```
expr    := term (('+' | '-') term)*
term    := factor (('*' | '/' | '%') factor)*
factor  := '-'? atom
atom    := INT | SYM | '(' expr ')' | func '(' expr (',' expr)* ')'
func    := 'ceildiv' | 'max' | 'min' | 'align'
INT     := [0-9]+
SYM     := [A-Za-z_][A-Za-z0-9_]*
```

### 2.2 求值语义

这部分必须逐条实现，不能"差不多就行"。

| 规则 | 定义 | 违反后果 |
|---|---|---|
| 整数类型 | 全程 `int64_t` | — |
| 溢出 | 溢出即错误（`LM_E_SHAPE`），不做回绕 | 回绕会导致静默的错误访存 |
| 值域 | 求值过程中所有值必须非负 | 出现负数即报错 |
| `/` | 截断除法；操作数非负，等价于向下取整 | — |
| `%` | 余数，符号随被除数；操作数非负，故结果非负 | — |
| `ceildiv(a,b)` | `a/b + (a%b != 0 ? 1 : 0)`，要求 `b > 0` | `b <= 0` 报错 |
| `max` / `min` | 变参，至少一个 | — |
| `align(a,b)` | `ceildiv(a,b)*b`，`b` 必须是 2 的幂 | `b` 非 2 的幂报错 |
| 未定义符号 | 报错并给出符号名 | 不允许默认为 0 |
| 短路 | 没有，表达式必须可以完全求值 | — |

要求值域非负，是因为 C 中 `-1 / 2 == 0` 和 `-1 % 2 == -1` 是常见陷阱。形状计算里一旦出现负数，通常意味着上游逻辑已经错了，静默继续只会把错误放大到访存层面。

`/` 的语义之所以必须是 S0：它出现在已经发布的包里。引擎若某天改成浮点或改变舍入方式，旧包算出的偏移就会变化，而且不会报错，只会踩内存。

### 2.3 为什么需要自造表达式

因为动态 shape。YOLO 的 batch 是 1 到 32，分辨率有若干档，通道数随版本变化。所有偏移都必须写成 `N`、`H`、`W` 的函数，否则每个 shape 都要一份 plan，包数量会爆炸。

---

## 3 Slot

```json
{
  "id": 7,
  "role": "temp",
  "dtype": "f16",
  "shape": ["N", "H", "W", "256"],
  "strides": null,
  "layout": 2,
  "elem_offset": 0,
  "alias_of": null,
  "align": 256,
  "rw": "rw"
}
```

| 字段 | 必需 | 语义 |
|---|---|---|
| `id` | 必需 | 唯一整数 |
| `role` | 必需 | `input`、`output`、`temp`、`param`、`persistent`、`scalar` |
| `dtype` | 必需 | 见 04 文档的 `LmDType` |
| `shape` | 必需 | `SymExpr[]`，空数组表示 rank 0 标量 |
| `strides` | 可选 | `SymExpr[]`，缺省为行主序紧凑布局，由引擎从 shape 计算 |
| `layout` | 可选 | 整数标签，缺省 0。引擎只比较/搬运标签，不解释语义；0 不是通配符，可读名在 `package.json.layouts` 中映射 |
| `elem_offset` | 可选 | 元素偏移，缺省 0 |
| `alias_of` | 可选 | 若存在，本 slot 是另一 slot 的视图 |
| `align` | 可选 | 缺省 256 |
| `rw` | 必需 | `r`、`w`、`rw`、`none` |

### 3.1 视图也是 slot

一个 concat node 的三个输入，在内存里可能是三块独立内存，也可能是一个大 buffer 的三个区段。

本规范选择把视图显式化成 slot：

```json
[
  {"id": 30, "role": "temp", "dtype": "f16", "shape": ["N","H","W","512"]},
  {"id": 31, "role": "temp", "dtype": "f16", "shape": ["N","H","W","256"],
   "alias_of": 30, "elem_offset": 0},
  {"id": 32, "role": "temp", "dtype": "f16", "shape": ["N","H","W","256"],
   "alias_of": 30, "elem_offset": "N*H*W*256"}
]
```

这样引擎的别名检查器能看到全部重叠关系，不需要理解 concat。如果视图藏在 node 里，引擎就必须知道"concat 的第 i 个输入偏移是多少"，那属于语义。

代价是 slot 数量变多。但 slot 由机器生成，数量不是问题，可验证性才是。

---

## 4 Node

```json
{
  "id": 17,
  "ku": "c2f_block",
  "impl": "sm90_f16_128x64_v3",
  "in":  [12, 41],
  "out": [13],
  "params": {"tile_m": 128, "tile_n": 64},
  "derived": {"M": "N*H*W", "K": "C"},
  "stream": 0,
  "after": [15],
  "flags": []
}
```

| 字段 | 必需 | 语义 |
|---|---|---|
| `id` | 必需 | 唯一整数，仅标识节点；不表达先后顺序 |
| `ku` | 必需 | KernelUnit 目录名 |
| `impl` | 必需 | 实现 id，对应 manifest 的 `implementations[].id` |
| `in` / `out` | 必需 | slot id 数组，顺序必须与该实现 manifest 的 `io` 一致 |
| `params` | 可选 | 覆盖 manifest 的默认参数，键必须在 schema 内 |
| `derived` | 可选 | 符号派生量，按键名排序后传给 kernel |
| `stream` | 可选 | 缺省 0；后续作为 schedule 的 stream 分配简写，首版非 0 必须拒绝 |
| `after` | 可选 | 额外的 happens-before 边。数据依赖由引擎自动推导 |
| `flags` | 可选 | 例如 `no_elim`，表示禁止死码消除 |

### 4.1 数据依赖由引擎推导

对首版的不可变值图：

1. 校验 node/slot id 唯一，所有引用存在。
2. 建立 slot → producer 表。外部输入与权重无图内生产者；其他 slot 恰有一个生产者。
3. 每个输入建立 producer → consumer 边；外部 slot 无需此边。节点不能读取自己正在产生的值。
4. 合并 `after` 前置边，去重，拒绝自依赖和环。
5. 以确定性拓扑排序得到默认提交顺序。节点在文件中的排列及 id 大小不决定数据版本。

后续若支持原地更新/alias，须先用逻辑值版本表达新旧数据，再结合底层存储区间校验 RAW、WAR、WAW；对共享状态等副作用用资源读写声明或显式顺序边表达。不能按 id 猜最近一次写入。声明不完整的任意二进制不能由图校验器保证无竞争。

### 4.2 可选 schedule

```json
"schedule": {"order": [17, 3, 9]}
```

`order` 必须恰好包含所有 node id 一次，且每条依赖的源出现在目标之前。未知 id、重复、遗漏或逆依赖均拒绝。无此字段时使用第 1 节的默认顺序。未来多流分配可以放在此对象内；当前节点的 `stream` 是草案简写，不能出现两处互相矛盾的分配。

主机提交顺序可确定，不代表不同 stream 的实际完成顺序或 kernel 内浮点归约一定确定。数值确定性需另行验证。

### 4.3 死码消除（后续可选）

如果某个 node 的所有输出都不被后续 node 读取、不是 plan 的 output、没有 `no_elim` 标记，且其 KernelUnit 未声明 `LM_KUC_SIDE_EFFECTS`，可以删除。

还必须保留 `after` 顺序约束和副作用；图改写后应重新生成并校验 schedule。首批原型不做删除，显式 schedule 覆盖全部节点。

---

## 5 Variant

这是处理形状组合的主要机制。

```json
"domain": ["N >= 1", "N <= 32", "H % 32 == 0", "W % 32 == 0"],

"variants": [
  {
    "id": "default",
    "when": "true",
    "note": "符号化 kernel，覆盖全域，性能次优但永远可用"
  },
  {
    "id": "hot_n1_640",
    "when": "N == 1 && H == 640 && W == 640",
    "overrides": {
      "17": {"impl": "sm90_f16_128x128_hardcoded"},
      "23": {"impl": "sm90_f16_64x64_hardcoded"}
    },
    "mem_plan": "mem/n1_640.json"
  },
  {
    "id": "hot_n4_640",
    "when": "N == 4 && H == 640 && W == 640",
    "overrides": {}
  }
]
```

### 5.1 三级形状策略

| 级别 | 形态 | 成本 | 使用场景 |
|---|---|---|---|
| L0 符号化 kernel | kernel 在运行时读 `dims`，用 `derived` 算网格与 tile | 一个 KernelUnit 覆盖所有 shape | 默认。多数 kernel 符号化只损失 5% 到 15% |
| L1 热形状特化 | 针对具体 shape 硬编码 tile，作为 override | 每个热 shape 一套 | 只对前三个形状做 |
| L2 形状类特化 | 针对形状区间单独写 kernel | 高 | 极少数，例如大 batch 需要 split-K |

纪律是：L0 必须存在且覆盖全域，L1 和 L2 是加速手段而不是必要条件。

某个 shape 没有对应 override 时回落到 L0，结果仍然正确。L0 由 Agent 编写，但验收标准是"全域正确且不慢于 ref 的三倍"，而不是"最快"。

这样形状组合不会爆炸：N 个 shape 只需要一套 L0 加三套 L1，而不是 160 套。

### 5.2 谓词语言

与 `SymExpr` 同源，但求值为布尔：

```
pred := expr (('<'|'<='|'=='|'!='|'>='|'>') expr) | 'true' | 'false'
      | pred ('&&' | '||') pred | '!' pred | '(' pred ')'
```

同样为 S0，因为它出现在已发布的包里，选择哪个 variant 会改变执行结果。

### 5.3 选择算法

算法必须是确定性的：

```
candidates = { v ∈ variants | eval(v.when) 为真 }
若 candidates 为空，返回 LM_E_SHAPE，并指出哪个域谓词失败
按 (特异性, 声明顺序) 排序
特异性 = override 数量 * 1000 + 谓词中的比较运算数量
取第一个
```

特异性必须可计算且写入文档，否则不同引擎版本会选出不同 variant，性能就不可复现。

---

## 6 内存计划

### 6.1 两种模式

| 模式 | 计算方 | 使用场景 |
|---|---|---|
| `"mem_plan": "auto"`，默认 | 引擎在形状绑定后做 liveness 加 first-fit | 起步阶段与非关键路径 |
| `"mem_plan": "explicit"` | Kernel Harness 离线计算，可以语义感知、可以全局最优；引擎只验证 | 性能敏感场景 |

### 6.2 生命周期与完成关系

单 stream 串行执行时，按选定 schedule 的位置计算 `first_use`、`last_use`，不能使用 node id。区间端点均包含，同一节点内的输入与输出不能默认共享存储。外部输入的借用期、输出的可见期以及 persistent 的会话生命周期需单独保留。

多 stream 时，CPU 提交位置不代表设备完成。两个独立 slot 只有在一方的全部访问完成 happens-before 另一方的首次写入，或反向关系成立时，才允许复用物理区间。无法证明时按同时存活处理，不复用。关系必须来自数据/显式依赖、同 stream 顺序及已插入的同步；不能先假定复用安全再反推依赖。

alias 是有意共享，必须额外验证区间、访问模式和同步；不能仅因声明了 alias 就跳过竞争检查。原型期不支持 alias 与多流。

### 6.3 explicit 模式的校验规则

在加载后、首次运行前，引擎对每个形状绑定做一次校验：

```
1. 对每个 slot，用检查溢出的算术验证 offset + size(s) <= arena.bytes
                                                       否则 LM_E_MEM_PLAN
2. 对物理区段相交的独立 slot，必须证明满足第 6.2 节的复用条件
                                                       否则 LM_E_MEM_PLAN
3. 对每个 alias 关系，子区段必须完全落在父区段内
4. 对齐：offset % align == 0
```

引擎能验证这些，是因为它们都是数据流性质。

### 6.4 auto 模式的分配算法

```
单 stream 按 schedule 中 first_use 的位置升序遍历 slot：
  候选区段为所有已释放的区段，按 offset 升序
  取第一个能容纳且满足对齐的
  都不行则在 arena 尾部追加
```

首版选择 first-fit 以减少实现量。其他算法也可以确定性实现，并不因使用 best-fit 就失去可复现性。多流分配须使用第 6.2 节的完成关系。

更好的分配器属于 explicit 模式，由 Kernel Harness 实现。

---

## 7 执行语义

一次 `lm_session_run` 的精确步骤：

```
1. bind     从 input buffer 的 dims 推导符号值，绑定关系在 plan.symbols 声明
2. domain   校验 plan.domain 的谓词全部为真，否则 LM_E_SHAPE，并指出失败谓词
3. variant  按第 5.3 节选择 variant
4. deps     构建数据边与 after 边，校验无环，生成或验证 schedule
5. mem      按 schedule 和完成关系计算或校验内存计划
6. alias    校验 slot 区间及访问冲突；原型不支持时拒绝
7. freeze   若启用 graph 模式且 key 命中，走重放
8. launch   按验证后的 schedule 提交；支持多流后按依赖插入 event
9. wait     在 output 所在 stream 插入 event，返回句柄或同步等待
10. return  返回结构化结果，包括耗时与是否降级
```

拓扑序由固定的就绪集合选择规则保证确定性。必须先完成依赖分析，再规划内存和提交；显式 schedule 也不能越过数据依赖。

---

## 8 序列化

第一版只用 JSON，便于生成、diff 和审计。

```json
{
  "plan_version": "1.0",
  "symbols": [
    {"name": "N", "bind": {"slot": 0, "dim": 0}},
    {"name": "H", "bind": {"slot": 0, "dim": 1}},
    {"name": "W", "bind": {"slot": 0, "dim": 2}}
  ],
  "domain": ["N >= 1", "N <= 32"],
  "slots": [],
  "nodes": [],
  "variants": [],
  "inputs": [0],
  "outputs": [13, 14, 15],
  "arena": {"dev": {"bytes": "align(ceildiv(N*H*W*640*2,256)*256 + 4096, 256)"}},
  "mem_plan": "auto",
  "meta": {"generator": "harness-emit/0.4.1", "model_hash": "sha256:...", "note": ""}
}
```

体积方面，YOLO 的 plan 约 60 个 node、200 个 slot，JSON 约 60 到 120 KB，加载耗时小于 1 ms，不需要二进制格式。若将来需要，`plan_format` 字段已经预留。

未知且不影响执行的元数据可忽略；影响调度、数值或内存语义的扩展必须声明版本/所需能力，不支持时拒绝，见 03 文档第 4 节。

---

## 9 完整示例

下面是 YOLOv8n head 的一部分：

```json
{
  "id": 31,
  "ku": "upsample_concat",
  "impl": "sm90_f16_symbolic",
  "in": [20, 29],
  "out": [30],
  "params": {"scale": 2, "mode": "nearest"},
  "stream": 0
},
{
  "id": 32,
  "ku": "c2f_block",
  "impl": "sm90_f16_symbolic",
  "in": [30, 41],
  "out": [31],
  "params": {"n": 1, "act": "silu"},
  "derived": {"M": "N*H*W", "K": "128"},
  "stream": 0
},
{
  "id": 33,
  "ku": "conv_bn_silu",
  "impl": "sm90_f16_128x128",
  "in": [31, 50],
  "out": [32],
  "params": {"stride": 2, "pad": 1, "k": 3},
  "stream": 0
}
```

node 31 的输入来自 backbone 的中间输出和上采样结果。引擎不需要知道这是特征金字塔，它只知道这两个 slot 必须在 node 31 之前写好。

---

## 10 常见错误

| 做法 | 问题 |
|---|---|
| 在 plan 里写 `"op_type": "conv"` | 语义泄漏 |
| 认为 node 顺序无所谓，交给引擎排 | 执行不可复现 |
| 用 `after` 表达数据依赖 | 引擎会自动推导，手写容易遗漏或重复 |
| 把 tile size 写死在 plan 里 | 应放在 variant override 或 kernel 内部 |
| 手写 `strides` | 引擎不校验其数学一致性，必须由合成器生成 |
| 视图 slot 漏写 `alias_of` | 引擎认为它们不重叠，会分配重叠内存 |
| 用浮点表达式算形状 | `SymExpr` 只有整数 |

---

## 11 为什么 plan 由程序生成

有人会问：既然 Kernel Harness 能产出 plan，它不就是一个编译器吗。

是，但它在引擎之外，而且它的"编译"是确定性的、可单元测试的、可 diff 的。

这也是对"引擎不懂语义就做不了全局优化"这一质疑的回答：全局优化（布局传播、buffer 复用、融合分组、形状特化）全部在 Kernel Harness 中完成，那里可以语义感知。引擎只做验证与执行。

因此 fusion-aware 是 Kernel Harness 的属性，不是引擎的属性。把它放回引擎，等于把易变的东西放回稳定层，会退回 TVM 的路线。这条分工见 01 文档的 P2 与 P15。
