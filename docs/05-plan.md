# 05 Plan 格式规范（v1，S1）

Plan 是程序映像：一份显式编排，说明用哪些 KernelUnit、按什么顺序、在哪个 stream、用哪块内存、参数是什么。

一个关键判断是：plan 不是编译出来的，是生成出来的。生成它的是 Kernel Harness 中的确定性合成器，不是 LLM，也不是引擎的某个 pass。

---

## 1 Plan 的角色

| 它是 | 它不是 |
|---|---|
| 一份显式的 kernel 调用序列与缓冲区绑定 | 一个等待优化的图 |
| 已经做完调度决策的产物 | 调度器的输入 |
| 引擎可以验证的对象 | 引擎可以改写的对象 |
| 由确定性合成器生成 | 由 Agent 自由创作 |

### 一切都用 plan 表达

这个统一性显著缩小了引擎的表面。

| 场景 | 常见引擎的做法 | 本项目 |
|---|---|---|
| 权重预打包（layout 转换、量化） | 专门的 prepack 代码路径 | 一个 plan，加载时执行一次 |
| 图像预处理 | 内建 preprocess 模块 | 一个 plan |
| 后处理（decode、NMS） | 内建 postprocess 模块 | 一个 plan |
| 主推理 | 图转 engine | 一个 plan |
| 校准与量化 | 独立工具 | Kernel Harness 的 plan，不进入运行时 |

引擎只有一种执行对象，没有特殊路径。这消掉了引擎里大量按场景分支的代码。

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
| `layout` | 可选 | 整数标签，缺省 0。引擎只搬运不解释，可读名在 `package.json.layouts` 中映射 |
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
| `id` | 必需 | 唯一整数。错误定位、日志、计时都以它为准 |
| `ku` | 必需 | KernelUnit 目录名 |
| `impl` | 必需 | 实现 id，对应 manifest 的 `implementations[].id` |
| `in` / `out` | 必需 | slot id 数组，顺序必须与该实现 manifest 的 `io` 一致 |
| `params` | 可选 | 覆盖 manifest 的默认参数，键必须在 schema 内 |
| `derived` | 可选 | 符号派生量，按键名排序后传给 kernel |
| `stream` | 可选 | 缺省 0 |
| `after` | 可选 | 额外的 happens-before 边。数据依赖由引擎自动推导 |
| `flags` | 可选 | 例如 `no_elim`，表示禁止死码消除 |

### 4.1 数据依赖由引擎推导

引擎不需要语义就能推导数据依赖，因为它只看 slot 的读写：

```
对每个 slot s：
  W(s) 为按顺序写 s 的 node 列表
  R(s) 为按顺序读 s 的 node 列表
  规则：R_i 必须 happens-after 最近一次写（RAW）
        下一次写必须 happens-after 前一次写，以及该写之前的全部读（WAW、WAR）
```

这是纯数据流推理，与"这个 node 是卷积还是 relu"无关。它带来三个收益：

1. Agent 写不出竞争条件。即使把有数据依赖的两个 node 放在不同 stream，引擎也会自动插入 event，或报 `LM_E_ALIAS`。
2. `after` 只需要声明非数据依赖，例如共享某个 scratch buffer 或人为指定的流水顺序，出错面更小。
3. 只要 plan 固定，执行顺序就固定。

这是"引擎可以在数据流上做完整正确性推理，但始终不知道节点在算什么"的具体体现。

### 4.2 死码消除

如果某个 node 的所有输出都不被后续 node 读取、不是 plan 的 output、没有 `no_elim` 标记，且其 KernelUnit 未声明 `LM_KUC_SIDE_EFFECTS`，可以删除。

这个判断只需要"输出有没有被读"，不需要语义。删除必须记录日志，并在 `--strict-plan` 下转为错误，提示生成器产出了无用节点。

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

### 6.2 生命周期定义

对 slot `s`：

```
first_use(s) = min{ i | node_i 读或写 s }    若 s 是 input，则为 0
last_use(s)  = max{ i | node_i 读或写 s }    若 s 是 plan output，则为 +∞
live(s)      = [first_use(s), last_use(s)]
overlap(a,b) 当且仅当 live(a) 与 live(b) 相交
```

output 和 persistent slot 在 plan 期间始终算活着。

生命周期精确到 node 序号而不是大致范围，因为差一个 node 就会踩内存，而这类问题最难排查。

### 6.3 explicit 模式的校验规则

在加载后、首次运行前，引擎对每个形状绑定做一次校验：

```
1. 对每个 slot，offset + size(s) <= arena.bytes        否则 LM_E_MEM_PLAN
2. 对每对 (a,b)，若 live 区间重叠且物理区段相交，且两者无 alias 关系
                                                       否则 LM_E_MEM_PLAN
3. 对每个 alias 关系，子区段必须完全落在父区段内
4. 对齐：offset % align == 0
```

引擎能验证这些，是因为它们都是数据流性质。

### 6.4 auto 模式的分配算法

```
按 first_use 升序遍历 slot：
  候选区段为所有已释放的区段，按 offset 升序
  取第一个能容纳且满足对齐的
  都不行则在 arena 尾部追加
```

不做 best-fit，不做重排，不做颜色分配。确定性的优先级高于最优性，因为需要可复现。

更好的分配器属于 explicit 模式，由 Kernel Harness 实现。

---

## 7 执行语义

一次 `lm_session_run` 的精确步骤：

```
1. bind     从 input buffer 的 dims 推导符号值，绑定关系在 plan.symbols 声明
2. domain   校验 plan.domain 的谓词全部为真，否则 LM_E_SHAPE，并指出失败谓词
3. variant  按第 5.3 节选择 variant
4. mem      auto 模式计算偏移，explicit 模式做校验
5. alias    校验所有 slot 的重叠关系合法
6. deps     构建 happens-before 图（数据边加 after 边），校验无环
7. freeze   若启用 graph 模式且 key 命中，走重放
8. launch   按 node id 升序提交，对跨 stream 依赖插入 event 等待与记录
9. wait     在 output 所在 stream 插入 event，返回句柄或同步等待
10. return  返回结构化结果，包括耗时与是否降级
```

第 8 步按 node id 升序而不是按拓扑序，是有意的。拓扑排序有多种合法结果，不确定；数据依赖保证正确性，id 顺序保证确定性。

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
  "outputs": [13, 14, 15],
  "arena": {"dev": {"bytes": "align(ceildiv(N*H*W*640*2,256)*256 + 4096, 256)"}},
  "mem_plan": "auto",
  "meta": {"generator": "harness-emit/0.4.1", "model_hash": "sha256:...", "note": ""}
}
```

体积方面，YOLO 的 plan 约 60 个 node、200 个 slot，JSON 约 60 到 120 KB，加载耗时小于 1 ms，不需要二进制格式。若将来需要，`plan_format` 字段已经预留。

未知字段必须被忽略，这是向前兼容的前提，见 03 文档第 4 节。

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
