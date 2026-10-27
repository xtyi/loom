# 17 实验 JSON plan 与目录包

当前可从目录加载静态执行图、入口契约、动态库路径和权重，并通过公共执行器选择 CPU/CUDA backend 执行。
本实现采用单独的实验格式，不声称实现 04–06 文档的完整 v1 草案，不冻结兼容性。

## 1 快速运行

在仓库根目录执行：

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DLOOM_ENABLE_EXPERIMENTAL_ABI=ON
cmake --build build -j4
./build/loom-cli inspect build/examples/cpu.loom
./build/loom-cli validate build/examples/cpu.loom
./build/loom-cli run build/examples/cpu.loom build/package-output \
  build/examples/cpu.loom/inputs/0.bin
```

构建时生成自包含的 `build/examples/cpu.loom/`：

```text
cpu.loom/
├── package.json              包入口、宿主要求、文件索引
├── plans/main.json           Graph、EntryBinding 和可选 schedule
├── kernels/add/ku.json       add 的 EntryContract 与 library 路径
├── kernels/add/libloom-example-add.so
├── kernels/relu_scale/ku.json
├── kernels/relu_scale/libloom-example-relu.so
├── weights/bias.bin          slot 1 的原生 f32 存储
└── inputs/0.bin              示例输入，不是运行时自动绑定的常量
```

库文件名按平台生成。JSON 模板在 [examples/packages/cpu](../examples/packages/cpu/)，
由 CMake 填入平台、大小端、指针位数、库后缀和参数编码；构建、加载均不依赖 Python。
整个生成目录可移动到任意位置，所有索引路径均相对于包根目录。

示例仍是 `relu(x + bias) * 2`。输入是 `[-3,-2,-1,0,1,2]`，bias 全为 1，
输出为 `[0,0,0,2,4,6]`，写入 `build/package-output/0.bin`。
输入文件按 `graph.inputs` 的顺序提供；输出文件按 `graph.outputs` 的位置命名，
文件内容包含该 tensor 的完整存储跨度（包括 stride padding）。已有输出文件会被拒绝覆盖。
可换一个输出目录再次运行。

默认构建也提供 `loom-cli inspect/validate`，但 `run` 要显式开启实验 ABI。
`inspect/validate` 检查元数据、目标兼容性、引用文件和权重内容大小，不加载或执行动态库，
也不证明 kernel 的数值行为。只有执行路径才调用现有二进制 caps 检查。

## 2 格式与检查顺序

三个 JSON 根对象的 `format` 分别必须精确等于：

| 文档 | format | 已识别的 required_capabilities |
|---|---|---|
| plan | `loom-plan-exp/1` | `static-tensors`、`single-stream` |
| kernel manifest | `loom-kernel-exp/1` | `static-tensors`、`cpu-sync` / `cuda-stream` |
| package | `loom-package-exp/1` | `static-tensors`、`single-stream`、`cpu-sync` / `cuda-stream` |

`required_capabilities` 是可选数组；未知能力拒绝。版本完全匹配，不猜测未来 minor/major
版本是否兼容。执行字段严格校验，未知字段拒绝；只有根级 `meta` 对象的内容不解释。
JSON 解析使用仓库内固定版本的 nlohmann/json 3.12.0，不自行实现解析器，第三方许可证
和来源见 [third_party/nlohmann](../third_party/nlohmann/README.md)。

供生成器和编辑器使用的 JSON Schema（2020-12）在 [schemas/experimental](../schemas/experimental/)。
运行时用类型解码器检查必需字段、字段类型和整数范围，再复用现有 tensor/graph/program 校验器。
Schema 描述文档结构；不能替代运行时对图依赖、对齐、跨度溢出、跨文件契约和宿主兼容性的检查。
整数位置在运行时必须使用 JSON 整数 token；`1.0`、布尔值和字符串均不自动转换。

加载顺序为：

1. 读取 `package.json`，检查格式、能力和目标要求。
2. 读取 plan，解码 Graph/EntryBinding，校验静态 tensor 与拓扑。
3. 读取各 manifest，解码 EntryContract，解析包内动态库路径。
4. 调用 `prepare_program()`，校验完整注册表、所有端口、参数长度和数值契约。
5. 检查常量映射完整且唯一，按 tensor 的存储跨度读取并对齐保存。
6. 执行时重新进入公共 Session：复验元数据、加载二进制、比对 caps、复制权重，再调用 `run()`。

失败抛出 `io::Error`，包含 `code/source/field`；原有 program 错误还保留 `Diagnostic`，
提供适用的 node/slot、expected/actual。没有部分成功的 Package 返回。

## 3 Plan：连接哪几个 slot、调用哪个实现

```json
{
  "format": "loom-plan-exp/1",
  "slots": [
    {"id": 0, "external": true, "tensor": {"dtype": "f32", "shape": [6], "alignment": 16}},
    {"id": 1, "external": false, "tensor": {"dtype": "f32", "shape": [6], "alignment": 16}}
  ],
  "nodes": [{
    "id": 10, "inputs": [0], "outputs": [1],
    "binding": {
      "kernel": "relu_scale", "implementation": "cpu",
      "numerical_contract": "f32-relu-scale/v1",
      "params_hex": "00000040"
    }
  }],
  "inputs": [0],
  "outputs": [1]
}
```

这是一个小端平台的单节点示意；完整可运行图在生成包的 `plans/main.json`。
字段直接对应当前 C++ 结构，不做自动 Cast、广播、shape 推导或 kernel 选择：

- slot 的 `id/external/tensor` 对应 `Slot`；文件中的 ID 原样保留，支持乱序、稀疏 ID。
  C++ 建图接口继续自动分配新 slot ID；加载的图仍可使用该接口继续编辑。
- tensor 必填 `dtype/shape`；`strides` 缺省为紧凑行主序，`layout` 缺省 0，`alignment` 缺省 1。
  支持当前 rank 0..8、零大小和正 stride 静态存储规则。
- node 必填 `id/inputs/outputs/binding`；可选 `after`；端口按数组位置绑定。
- `binding` 的 kernel/implementation 是注册键；`numerical_contract` 是节点要求。
  `params_hex` 缺省为空，以两个 hex 字符表示一个原始字节，不按字段解读参数。
  小端 `00000040` 是原生 f32 的 2.0，大端对应 `40000000`，由打包工具编码。
- 图级 `inputs` 是调用接口，其他 external slot 必须由包的 constants 覆盖；`outputs` 保留顺序和重复项。
- 可选 `schedule: {"order": [30,10]}` 必须恰好覆盖全部 node 且满足依赖；
  未指定时沿用最小就绪 ID 的确定性拓扑排序。显式空数组不等于未指定。

`io::parse_plan(text)` 与 `io::load_plan(path)` 只解码 plan、检查 tensor 和拓扑。
没有独立的入口注册表时，不能仅凭它们声称完整契约已验证；跨文件检查由 `load_package()` 完成。

## 4 Manifest：实现承诺的入口契约

```json
{
  "format": "loom-kernel-exp/1",
  "kernel": "relu_scale", "implementation": "cpu",
  "symbol": "lm_ku_entry_exp_v1",
  "inputs": [{"name": "x", "tensor": {"dtype": "f32", "shape": [6], "alignment": 16}}],
  "outputs": [{"name": "y", "tensor": {"dtype": "f32", "shape": [6], "alignment": 16}}],
  "numerical_contract": "f32-relu-scale/v1",
  "params_bytes": 4,
  "workspace_bytes": 16,
  "workspace_alignment": 16,
  "library": "kernels/relu_scale/libloom-example-relu.so"
}
```

每个文件对应一个 EntryContract 和一个实现路径。`params_bytes/workspace_bytes` 缺省 0，
`workspace_alignment` 缺省 1；其他上面列出的字段必填。
同一入口可以被多个 node 调用，但包中不允许重复注册或提供完全未使用的入口。
`EntryBinding.numerical_contract == EntryContract.numerical_contract`，
执行加载时还检查 `EntryContract == binary caps` 的相关声明。

## 5 Package：把文件组织为运行单元

`package.json` 必填 `format/name/target/plan/kernels/constants`。
一个 plan、若干 kernel manifest、每个常量 slot 一个原始 `.bin` 文件。
`target` 必填 `backend/abi/os/arch/endianness/pointer_bits`，backend 为 `cpu` 时 abi 为 `loom.cpu.exp.v1`；backend 为 `cuda` 时 abi 为 `loom.cuda.exp.v1`，并须填写 `compute_capability`（例如 120）。其余字段必须精确匹配当前宿主。

`compute_capability` 在创建 CUDA Session 时与选定 GPU 精确匹配。CPU 包不得声明该字段。元数据加载不探测 GPU，因此无 CUDA 构建仍可 inspect/validate CUDA 包；真正运行时必须启用 CUDA backend。`required_capabilities` 的 `cpu-sync` / `cuda-stream` 必须与包目标一致，kernel manifest 也执行相同检查。
已识别 arch 为 `x86_64/aarch64/x86/arm`，os 为 `linux/macos/windows`；
CPU/CUDA 动态库加载仍受现有 POSIX 限制。CUDA 示例包、构建和调用见 [18 执行器与后端](18-backends.md)。

所有 manifest 中的路径也相对于包根目录，禁止绝对路径、`..`、反斜杠和冒号。
解析符号链接后仍须位于包内且为普通文件。常量文件大小必须恰好等于 slot 的 `span_bytes`，
允许零跨度对应空文件；常量以满足 slot 对齐的内存保存。
多次常量映射可以引用同一个文件，最终读取为独立的 slot 存储。

加载器只做粗粒度宿主兼容检查，不检查 CPU 指令集扩展、编译器/系统库版本，也不执行二进制架构审计。
这是运行可信、配套构建的本地 kernel 的实验目录包；包在加载和执行期间应保持不变。
当前没有归档解包、指纹、内容哈希、签名、registry 或隔离执行。
JSON 文档限制为 16 MiB、64 层嵌套，重复对象键、注释和尾随逗号拒绝。

## 6 宿主接入

```cpp
#include "cpu/package.h"

auto session = loom::cpu::load_package("build/examples/cpu.loom");
alignas(16) float x[] = {-3,-2,-1,0,1,2};
auto result = session->run({{x, sizeof(x)}});
```

需要先 inspect 再执行时，可使用 `io::load_package()` 得到元数据和常量，
再交给 `cpu::make_session(package)`。返回的 Session 复制自己的元数据与常量，
Package 销毁后仍可运行；输出沿用现有 RunResult 的独立所有权。

当前仍不支持动态 shape、SymExpr、variant、alias、原地更新、arena、多流、设备后端、
自动数值校验或目标 S0 ABI。新增这些语义时，需要扩展并重新编号实验格式。
