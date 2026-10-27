# 16 实现起点

当前是验证执行契约的原型，已跑通实验 JSON 目录包加载、C ABI 与 CPU/CUDA 动态库的同步宿主执行，不是完整推理引擎。目标 v1 package、S0 ABI、校验服务等仍未实现。S0/S1 是未来发布目标，不能据此依赖当前内部类型的布局。新增目录包的精确格式与命令见 [17 JSON 与目录包](17-json-package.md)。

## 1 语言与模块

| 部分 | 选择 | 理由 |
|---|---|---|
| runtime / CLI | C++17 | 资源生命周期、设备接口、厂商库生态与本地调用 |
| 宿主与 KernelUnit 边界 | 实验 C ABI，CPU 同步 / CUDA stream 子集 | 隔离 C++ 对象布局；版本与能力检查；尚未冻结 |
| 构建与基础测试 | CMake / CTest | 无 GPU、无网络依赖即可运行 |
| 离线构建与验证 | Python，后续实现 | 模型生态、Agent 驱动、数据与实验流程 |
| Python 绑定 | 可选 pybind11，后续实现 | 包装宿主 API，核心不依赖解释器 |

详细理由与异步生命周期约束见 [决策 0010](decisions/0010-cpp-python.md)。

## 2 当前代码

plan 与宿主接口为内部 S3 C++ 类型，KernelUnit 使用单独的实验 S2 C ABI；JSON 使用单独编号的 `loom-*-exp/1` schema，均不是已发布的稳定格式。各层独立验证：

| 入口 | 作用 | 成功结果 |
|---|---|---|
| `prepare(graph, order)`，`plan/graph.h` | 只检查拓扑与 schedule，忽略 tensor/入口元数据 | 确定性提交顺序 |
| `validate_tensor(desc)`，`plan/tensor.h` | 检查静态存储描述和尺寸溢出 | 规范化 descriptor、元素数与存储跨度 |
| `prepare_program(graph, entries, order)`，`plan/program.h` | 拓扑、全部 tensor、入口注册表与每个端口绑定 | 拥有自身数据的 PreparedProgram 快照 |
| `cpu::Session(graph, entries, libraries, constants, order)` | 重新 prepare、加载使用到的动态库、匹配二进制 caps、复制常量 | 拥有 metadata、常量与动态库句柄的 Session |
| `session.run(inputs)` | 输入容量/对齐/重叠检查，分配 buffer/workspace，按顺序同步调用 | 拥有输出存储的 RunResult |
| `io::load_plan(path)` | 严格 JSON 解码、静态 tensor 与拓扑校验 | 保留文件 ID 的 Graph 和可选 order |
| `io::load_package(directory)` | 版本、能力、宿主要求、跨文件契约与常量加载 | 可 inspect 的 Package，不加载动态库 |
| `cpu::load_package(directory)` | 包加载后交给公共 Session | 独立拥有数据并检查二进制 caps 的 Session |

Slot 使用独立 id；外部 slot 已初始化且只读，其余 slot 恰有一个生产者。图级 `inputs` 明确每次调用传入的 slot 及其顺序：引用必须存在、不重复且为 external；不列入 inputs 的 external slot 可由加载阶段提供，如权重/常量。允许空输入列表、未消费的输入，以及输入直接作为图输出；不从 external 自动推断调用接口。节点按 `(kernel, implementation)` 绑定注册表里的 EntryContract。先建立 producer → consumer 与 `after` 边，再用最小就绪 id 拓扑排序；显式顺序必须覆盖全部节点且满足依赖。

纯 `prepare` 成功不意味着入口可调用。完整准备路径要求每个 slot 有有效描述、每个 node 有匹配入口，即使某些节点或 slot 未被输出消费，也不会跳过校验。失败时不返回部分 PreparedProgram；诊断包含可用的 node/slot、字段路径，以及匹配失败时的 expected/actual。

### 2.1 Graph 创建与修改

`Graph` 保留原始数据字段，同时提供轻量编辑 API。slot ID 自动分配并返回，调用方用局部变量命名 slot；node ID 仍显式指定，避免依赖嵌套初始化列表的字段顺序：

```cpp
loom::plan::Graph graph;
const auto input = graph.add_input();
const auto hidden = graph.add_slot();
const auto output = graph.add_slot();

graph.add_node(10).reads({input}).writes({hidden});
graph.add_node(20).reads({hidden}).writes({output});
graph.set_outputs({output});

// 修改已有节点：替换整个输入列表，也可修改 writes、after 和 bind。
graph.node(20).reads({input});
const auto result = loom::plan::prepare(graph);
```

`add_input` 创建 external slot 并追加到图输入；`add_external` 创建权重/常量等外部 slot，但不追加到图输入；`add_slot` 创建节点产生的 slot。三者均不接收 ID，可直接传入 TensorDesc，例如 `graph.add_slot(tensor)`；省略描述适用于仅演示拓扑，`prepare_program` 仍要求有效 tensor 和入口绑定。

ID 在空图中从 0 开始，否则取当前已有 slot 的最大 ID 加 1，因此直接填充原始字段后也可继续添加；最大 ID 无法递增时抛 `std::overflow_error`，不修改图。删除 slot 后，其 ID 可能被复用；调用方应使用每次返回的 ID，不依赖固定数值。

`set_tensor` 替换指定 slot 的描述，`set_inputs` / `set_outputs` 替换有序图接口。`bind` 的完整用法见 [contract_demo.cpp](../examples/contract_demo.cpp)。

编辑 API 检查重复 node ID、引用是否存在、写 external slot 等局部错误。因此 slot 必须先创建，再传给 reads/writes；after 引用的节点也必须已创建。缺失 ID 抛 `std::out_of_range`，其他非法编辑抛 `std::invalid_argument`。单次 setter 在检查通过后才替换字段；一整条链式调用不是事务，后一步失败不会回滚前面的操作。编辑期间允许图暂时不完整，环、跨节点多生产者等全图约束仍由 `prepare` 检查，tensor/入口契约仍由 `prepare_program` 检查。

`node(id)` 返回按 ID 查找的 NodeEditor；添加其他节点导致容器扩容，不会使它失效。使用期间 Graph 必须存活且未被移动。删除目标节点后编辑会报错；重新创建相同 ID 后，旧编辑器会选中这个新节点。

`remove_slot` 拒绝删除仍被节点或图接口引用的 slot。`remove_node` 拒绝删除仍被其他节点的 after 引用的节点，成功时仅移除该节点，保留其输出 slot；调用方随后需要补上生产者或清理相关连接和 slot，再做 prepare。删除不会级联修改其他节点。

### 2.2 TensorDesc

| 字段 | 当前语义 |
|---|---|
| `dtype` | f16、bf16、f32、f64、i8、u8、i32、i64、bool 的存储描述；bool 为 1 字节。无效/未知值拒绝 |
| `shape` | 静态非负整数，rank 0..8；空 shape 是一个标量，任一维为 0 则元素数/跨度为 0 |
| `strides` | 元素单位；空数组表示紧凑行主序，生成时用 `max(dim, 1)`；显式数组必须与 rank 相同且全为正 |
| `layout` | 不透明整数，精确匹配；0 是默认标签，不是通配符，不推导 NCHW/NHWC 语义 |
| `alignment` | 保证的基址字节对齐，非零 2 的幂；默认 1，不隐式推导自然对齐 |

dtype 被识别只表示知道存储宽度，不表示已实现该类型的算术 kernel。描述中不含 data 指针；公共 Session 在调用/分配阶段检查声明容量、地址对齐、跨输入 slot 的存储跨度重叠，并管理输出寿命。调用方仍须提供真实有效的主机内存及容量；设备指针不属于该接口。

非空 tensor 的 `span_bytes = (1 + sum((dim-1)*stride)) * element_bytes`，包含中间 padding，不包含尾部对齐补齐。标量占一个元素。计算使用检查过的乘法/加法，要求元素数、stride、存储跨度不超过 `min(SIZE_MAX, INT64_MAX)`；不分配该大小的内存。

非重叠证明按 stride 升序检查维度（忽略大小为 1 的维度）：每一维 stride 必须不小于此前维度可访问的跨度。支持常规 padding、切片步长和转置排列，但会保守拒绝某些特殊交错布局。零大小 tensor 无实际访问，仍检查 dtype、rank、每维范围、对齐和 stride 合法性；若其默认 stride 无法表示，也拒绝。

当前不支持广播 stride=0、负 stride、动态 shape、子字节打包、偏移和 alias。每个 slot 描述独立的基址；未来的视图契约需单独扩展。

### 2.3 EntryContract 与绑定

`plan/entry.h` 定义入口元数据：

- `(kernel, implementation)` 是唯一注册键，`symbol` 是需要加载的符号名；prepare 阶段仅检查名称非空且无 NUL，不解析库、不查找函数。公共 Session 随后按这个符号解析真实入口。
- `inputs`/`outputs` 是有序端口；输入只读、输出只写。每个方向内端口名称唯一，端口采用静态 TensorDesc。
- `numerical_contract` 是必要的、不透明的版本/内容 ID；节点必须明确要求同一 ID。它用于区分相同存储 dtype 下的不同累加/舍入方案，不证明二进制实际符合这些行为。
- `params_bytes` 要求节点提供恰好相同大小的参数块；内容不透明，尚不检查参数内部 schema。
- `workspace_bytes` 和 `workspace_alignment` 是固定资源要求；prepare 检查尺寸与对齐合法性，随后保存在调用快照中。公共 Session 每次 run 分配满足所有调用要求的共享 scratch，串行复用。

注册表中的所有入口均校验，包括未使用项。每个调用按位置逐端口比较 dtype、完整 shape、规范化 stride 和 layout，要求 slot 对齐保证不低于入口要求。省略 stride 与显式紧凑 stride 等价；包括 singleton 维在内的不同显式 stride 仍保守要求精确匹配。没有自动 Cast、布局转换、广播或 shape 推导。

相同描述的两个输入被调换时，仅凭类型无法发现语义错误；这仍需确定性的图生成器和数值验证。数值契约 ID 的内容定义、真实性及验证报告也由外部构建/验证流程负责。

### 2.4 结果与当前边界

PreparedProgram 保存规范化 tensor、external 标记、有序图级 inputs、按 schedule 排列的入口/输入输出/参数快照及图输出。修改原图或注册表不会影响已经准备好的结果。它只是内部元数据；尚未完成目标设备、ABI、动态库、地址、容量或数值行为验证，不能直接执行。

plan 层仍不执行 kernel、不分配 tensor/workspace。执行路径由下面的公共 Session 提供；JSON 与包加载由独立 `io/` 模块提供。当前不支持原地更新、状态、多流、控制流或动态输出。

### 2.5 实验 C ABI 与公共 Session

[experimental.h](../runtime/abi/experimental.h) 可供 C11 与 C++17 编译器使用，结构体只含固定宽度整数、固定数组和原生指针；运行时与动态库必须使用兼容的目标架构、调用约定和默认结构体布局。它使用 `lm_ku_caps_exp_v1` 能力查询符号和 EntryContract 指定的入口（示例为 `lm_ku_entry_exp_v1`），与 04 文档的目标 S0 ABI 不互通。实验版本号必须精确匹配；`struct_size` 至少覆盖本版结构。ABI 变更需重新编译库。

Session 创建时先调用 `prepare_program` 保存独立快照，不接受调用方任意构造的 PreparedProgram 作为已校验凭据。随后通过 `(kernel, implementation)` 匹配显式库路径，用 `dlopen(RTLD_NOW | RTLD_LOCAL)` / `dlsym` 加载实际入口。缺失、重复、未使用的库映射被拒绝。动态库必须声明所选 backend 的协议：CPU 为 SYNCHRONOUS，CUDA 为 STREAM_ORDERED；backend/flags 不匹配、未知能力位、版本不符、缺失符号都会被拒绝。二进制 caps 的名称、实现、入口符号、数值契约、端口数量、参数字节数、workspace 大小/对齐必须与 EntryContract 一致，各端口的对齐声明还须覆盖二进制最低要求。

完整 tensor 类型/shape/stride/layout 由已有注册表与图匹配；本版 caps 不重复存储端口 TensorDesc，也不证明库实际满足数值行为。二进制与元数据须配套提供；示例 kernel 还会自行检查可支持的 buffer 描述。动态库在宿主进程内运行，遵守 ABI 的本地代码是此原型的前提。

Buffer 与资源生命周期：

- `constants` 按 slot ID 传入，必须恰好覆盖图输入以外的全部 external slot；创建时检查、分配并复制，不依赖调用方后续保留权重。
- `run(inputs)` 按 graph.inputs 顺序接收 BufferView。输入借用到 run 返回，期间调用方不能修改；检查非空存储的非空地址、容量、对齐和地址范围溢出。零存储跨度允许 NULL。不同输入 slot 的存储跨度重叠被拒绝；构造时不同常量 slot 也执行相同检查。
- 每次 run 为全部内部 slot 独立分配满足对齐的内存，没有 arena、跨次 buffer 复用或生命周期复用。内部存储先清零，避免带 padding 的输出暴露未初始化字节；kernel 仍负责写全所有逻辑输出值。
- workspace 取所有调用的最大大小和最大对齐，每次 run 分配一次、按串行顺序复用；传给各 kernel 的容量为该入口声明值，内容不保证初始化。参数是只读字节块，不承诺类型对齐，kernel 用 memcpy 读取。
- 所有输入检查、buffer/workspace 和调用描述分配在执行第一个节点前完成。CPU stream 为 NULL；CUDA launcher 必须把所有工作提交到传入的 stream，公共执行器逐节点等待完成。两者均不得保留宿主描述/params 指针或跨 ABI 抛异常。
- RunResult 的 outputs 按图输出顺序保存独立于 Session 生命周期的共享所有权；CPU 内部结果直接保留分配，输入/常量直接作为输出时复制存储；CUDA 输出在返回前下载到宿主内存。重复的输出 ID 共享同一份结果；下次 run 不会覆盖旧结果。
- Session 销毁或构造失败时释放已加载的库与常量；每次调用的中间存储和 scratch 在返回/异常时释放。暂不承诺同一 Session 并发调用安全。

错误通过 `execution::Error` 抛出（`cpu::Error`、`nv::Error` 是兼容别名），包含错误类别、适用的 node/slot 和原始 kernel status；prepare 失败还保留原始 Diagnostic。kernel 返回 1..255 归为 kernel 错误，其他非零返回值归为协议错误；立即停止后续节点，不返回部分结果，也不回滚已经执行的 kernel 副作用。宿主内存分配失败使用标准 `std::bad_alloc`；CUDA API 错误归为 `ErrorCode::device`，设备执行错误带 node。

公共执行层与 backend 的职责、CUDA 设备资源、stream 收尾和构建命令见 [18 执行器与后端](18-backends.md)。

## 3 构建与验证

需要支持 C++17 的编译器与 CMake 3.20 或更新版本，默认与 CPU 构建不需要 Python、GPU SDK 或联网下载依赖；可选 CUDA 构建需要 CUDA Toolkit。JSON 解析器 nlohmann/json 3.12.0 及许可证已固定包含在 `third_party/`，只供 `loom_io` 内部使用。

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
./build/loom-plan-demo
./build/loom-contract-demo
```

实验 CPU 路径默认关闭；在 POSIX 平台显式开启，需要额外的 C11 编译器：

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DLOOM_ENABLE_EXPERIMENTAL_ABI=ON
cmake --build build
ctest --test-dir build --output-on-failure
./build/loom-cpu-demo ./build/libloom-example-add.so ./build/libloom-example-relu.so
./build/loom-cli inspect build/examples/cpu.loom
./build/loom-cli validate build/examples/cpu.loom
./build/loom-cli run build/examples/cpu.loom build/package-output build/examples/cpu.loom/inputs/0.bin
```

上面的库后缀适用于 Linux；其他平台用实际生成的文件名。示例输出 `relu((input + bias)) * 2: 0 0 0 2 4 6`，并逐元素检查结果。`add.c` 用 C 编译，`relu_scale.cpp` 用 C++ 编译，都是独立共享库，只包含实验 ABI 头文件、不链接 loom_cpu/loom_plan；构建时生成，运行时只加载。两个示例仅实现 scalar/1D f32，支持正 stride 与空 tensor，不代表 runtime 只接受这一种 dtype/shape。宿主调用见 [cpu_demo.cpp](../examples/cpu_demo.cpp)，图与入口声明见 [cpu_graph.h](../examples/cpu_graph.h)。

也可以在仓库根目录用外部编译命令替换 add 库，验证运行时不依赖其构建方式：

```sh
cc -std=c11 -fPIC -fvisibility=hidden -shared -Iruntime examples/kernels/add.c -o build/libexternal-add.so
./build/loom-cpu-demo ./build/libexternal-add.so ./build/libloom-example-relu.so
```

内存与未定义行为检查用独立构建目录开启，GCC 或 Clang 均可：

```sh
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug \
  -DLOOM_ENABLE_EXPERIMENTAL_ABI=ON -DLOOM_ENABLE_SANITIZERS=ON
cmake --build build-asan
ctest --test-dir build-asan --output-on-failure
```

该选项为全部目标（含动态库 kernel）加上 AddressSanitizer、UndefinedBehaviorSanitizer 和 LeakSanitizer，并启用 `-fno-sanitize-recover=all`，首次命中即失败。kernel 库一起插桩，所以越界访问会在库内被检出，而不是只依赖红色区域。在此配置下运行相同的 CTest 测试集。

CMake 默认导出编译数据库；使用 Ninja 或 Unix Makefiles 配置后，会生成 `build/compile_commands.json`，供 clangd 读取。

VS Code 打开仓库后，安装工作区推荐的 clangd 和 CMake Tools 扩展，并确保 `clangd` 在 `PATH` 中可用。工作区配置会在打开项目时运行 CMake，并让 clangd 使用 `build/` 下的编译数据库。也可以手动运行上面的配置命令，或执行 `CMake: Configure`；首次生成数据库后若编辑器诊断未刷新，可执行 `clangd: Restart language server`。已安装 Microsoft C/C++ 扩展时，工作区会关闭它的 IntelliSense，避免与 clangd 重复诊断。

`loom-plan-demo` 将乱序存放的四个节点准备为 `40 10 20 5`：生产者 40 必须先执行，两个就绪分支按 10、20 排列，最后汇合到 5。它说明节点 id 不等于执行次序。

`loom-contract-demo` 准备 FP32 → 显式 Cast → FP16 → 声明 FP32 累加的 KernelUnit 调用链。输出 slot 存储大小、符号名、数值契约及 workspace 要求；它不会实际进行类型转换或计算。

测试覆盖 Graph 创建/修改/删除、非法编辑拒绝、编辑器在容器扩容后的有效性，以及图级输入合法性、输入顺序与权重区分、DAG 结构、尺寸与跨度溢出、标量/空 tensor、padding/转置、非法 stride、入口重复/缺失、参数/端口数量、端口顺序、dtype/shape/layout/stride/对齐、数值契约以及快照所有权。使用实际检查而非依赖 `assert`，Release 构建也执行断言。

开启实验选项后，还测试真实动态库执行、C/C++ ABI 互通、权重快照、重复运行和输出寿命、输入顺序/容量/对齐/别名、scalar/空/带 stride 计算，以及通过故障动态库验证缺失符号、截断 caps、错误版本/能力/契约和 kernel 失败后停止。默认 Debug 构建与 `LOOM_ENABLE_SANITIZERS` 构建都执行同一批测试；后者额外检查越界、泄漏和未定义行为，当前均通过。

## 4 接下来的代码顺序

1. 静态 JSON schema、目录包加载、版本/能力拒绝规则和 CPU 文件执行闭环已实现；后续扩展 SymExpr、variant 与 arena/buffer 复用，逐步验证目标 v1 格式。
2. CPU/CUDA 公共执行闭环已完成（见 [18 执行器与后端](18-backends.md)）；后续增加显式 FP32/FP16 转换、混合累加契约和跨节点异步调度，再评审 ABI 冻结。
3. session API 完成上述验证后接入可选 Python 绑定；自动精度搜索、量化和高级调度单独扩展。

每一步先完成可验证的调用闭环，再增加范围。首批原型不等于 M0 全部完成。
