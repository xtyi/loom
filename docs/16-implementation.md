# 16 实现起点

当前是验证执行契约的原型，不是完整推理引擎。设计文档中的 package、ABI、后端、校验服务等多数仍未实现。S0/S1 是未来发布目标，不能据此依赖当前内部类型的布局。

## 1 语言与模块

| 部分 | 选择 | 理由 |
|---|---|---|
| runtime / CLI | C++17 | 资源生命周期、设备接口、厂商库生态与本地调用 |
| 宿主与 KernelUnit 边界 | C ABI，后续实现 | 隔离 C++ 编译器与对象布局；版本化接口 |
| 构建与基础测试 | CMake / CTest | 无 GPU、无网络依赖即可运行 |
| 离线构建与验证 | Python，后续实现 | 模型生态、Agent 驱动、数据与实验流程 |
| Python 绑定 | 可选 pybind11，后续实现 | 包装宿主 API，核心不依赖解释器 |

详细理由与异步生命周期约束见 [决策 0010](decisions/0010-cpp-python.md)。

## 2 当前代码

所有接口为内部 S3 C++ 类型，不是已发布的 C ABI 或 JSON schema。各层独立验证：

| 入口 | 作用 | 成功结果 |
|---|---|---|
| `prepare(graph, order)`，`plan/graph.h` | 只检查拓扑与 schedule，忽略 tensor/入口元数据 | 确定性提交顺序 |
| `validate_tensor(desc)`，`plan/tensor.h` | 检查静态存储描述和尺寸溢出 | 规范化 descriptor、元素数与存储跨度 |
| `prepare_program(graph, entries, order)`，`plan/program.h` | 拓扑、全部 tensor、入口注册表与每个端口绑定 | 拥有自身数据的 PreparedProgram 快照 |

Slot 使用独立 id；外部 slot 已初始化且只读，其余 slot 恰有一个生产者。图级 `inputs` 明确每次调用传入的 slot 及其顺序：引用必须存在、不重复且为 external；不列入 inputs 的 external slot 可由加载阶段提供，如权重/常量。允许空输入列表、未消费的输入，以及输入直接作为图输出；不从 external 自动推断调用接口。节点按 `(kernel, implementation)` 绑定注册表里的 EntryContract。先建立 producer → consumer 与 `after` 边，再用最小就绪 id 拓扑排序；显式顺序必须覆盖全部节点且满足依赖。

纯 `prepare` 成功不意味着入口可调用。完整准备路径要求每个 slot 有有效描述、每个 node 有匹配入口，即使某些节点或 slot 未被输出消费，也不会跳过校验。失败时不返回部分 PreparedProgram；诊断包含可用的 node/slot、字段路径，以及匹配失败时的 expected/actual。

### 2.1 Graph 创建与修改

`Graph` 保留原始数据字段，同时提供轻量编辑 API。调用方显式指定 ID，用局部变量命名 slot，避免依赖嵌套初始化列表的字段顺序：

```cpp
loom::plan::Graph graph;
const auto input = graph.add_input(0);
const auto hidden = graph.add_slot(1);
const auto output = graph.add_slot(2);

graph.add_node(10).reads({input}).writes({hidden});
graph.add_node(20).reads({hidden}).writes({output});
graph.set_outputs({output});

// 修改已有节点：替换整个输入列表，也可修改 writes、after 和 bind。
graph.node(20).reads({input});
const auto result = loom::plan::prepare(graph);
```

`add_input` 创建 external slot 并追加到图输入；`add_external` 创建权重/常量等外部 slot，但不追加到图输入；`add_slot` 创建节点产生的 slot。三者均可传入 TensorDesc；省略描述适用于仅演示拓扑，`prepare_program` 仍要求有效 tensor 和入口绑定。`set_tensor` 替换指定 slot 的描述，`set_inputs` / `set_outputs` 替换有序图接口。`bind` 的完整用法见 [contract_demo.cpp](../examples/contract_demo.cpp)。

编辑 API 检查重复 ID、引用是否存在、写 external slot 等局部错误。因此 slot 必须先创建，再传给 reads/writes；after 引用的节点也必须已创建。缺失 ID 抛 `std::out_of_range`，其他非法编辑抛 `std::invalid_argument`。单次 setter 在检查通过后才替换字段；一整条链式调用不是事务，后一步失败不会回滚前面的操作。编辑期间允许图暂时不完整，环、跨节点多生产者等全图约束仍由 `prepare` 检查，tensor/入口契约仍由 `prepare_program` 检查。

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

dtype 被识别只表示知道存储宽度，不表示已实现该类型的算术 kernel。描述中不含 data 指针；真实分配长度、地址对齐、设备、对象寿命和跨 slot 物理重叠仍需在调用/分配阶段检查。

非空 tensor 的 `span_bytes = (1 + sum((dim-1)*stride)) * element_bytes`，包含中间 padding，不包含尾部对齐补齐。标量占一个元素。计算使用检查过的乘法/加法，要求元素数、stride、存储跨度不超过 `min(SIZE_MAX, INT64_MAX)`；不分配该大小的内存。

非重叠证明按 stride 升序检查维度（忽略大小为 1 的维度）：每一维 stride 必须不小于此前维度可访问的跨度。支持常规 padding、切片步长和转置排列，但会保守拒绝某些特殊交错布局。零大小 tensor 无实际访问，仍检查 dtype、rank、每维范围、对齐和 stride 合法性；若其默认 stride 无法表示，也拒绝。

当前不支持广播 stride=0、负 stride、动态 shape、子字节打包、偏移和 alias。每个 slot 描述独立的基址；未来的视图契约需单独扩展。

### 2.3 EntryContract 与绑定

`plan/entry.h` 定义入口元数据：

- `(kernel, implementation)` 是唯一注册键，`symbol` 是未来需要加载的符号名；目前仅检查名称非空且无 NUL，不解析库、不查找函数。
- `inputs`/`outputs` 是有序端口；输入只读、输出只写。每个方向内端口名称唯一，端口采用静态 TensorDesc。
- `numerical_contract` 是必要的、不透明的版本/内容 ID；节点必须明确要求同一 ID。它用于区分相同存储 dtype 下的不同累加/舍入方案，不证明二进制实际符合这些行为。
- `params_bytes` 要求节点提供恰好相同大小的参数块；内容不透明，尚不检查参数内部 schema。
- `workspace_bytes` 和 `workspace_alignment` 是固定资源要求；检查尺寸与对齐合法性，随后保存在调用快照中。还没有分配或检查实际 workspace。

注册表中的所有入口均校验，包括未使用项。每个调用按位置逐端口比较 dtype、完整 shape、规范化 stride 和 layout，要求 slot 对齐保证不低于入口要求。省略 stride 与显式紧凑 stride 等价；包括 singleton 维在内的不同显式 stride 仍保守要求精确匹配。没有自动 Cast、布局转换、广播或 shape 推导。

相同描述的两个输入被调换时，仅凭类型无法发现语义错误；这仍需确定性的图生成器和数值验证。数值契约 ID 的内容定义、真实性及验证报告也由外部构建/验证流程负责。

### 2.4 结果与当前边界

PreparedProgram 保存规范化 tensor、external 标记、有序图级 inputs、按 schedule 排列的入口/输入输出/参数快照及图输出。修改原图或注册表不会影响已经准备好的结果。它只是内部元数据；尚未完成目标设备、ABI、动态库、地址、容量或数值行为验证，不能直接执行。

当前不执行 kernel、不解析 JSON、不分配 tensor/workspace 内存，不支持原地更新、状态、多流、控制流或动态输出。公共 C ABI 与 loader 仍是下一步。

## 3 构建与验证

需要支持 C++17 的编译器与 CMake 3.20 或更新版本，不需要 Python、GPU SDK 或下载第三方包。

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
./build/loom-plan-demo
./build/loom-contract-demo
```

CMake 默认导出编译数据库；使用 Ninja 或 Unix Makefiles 配置后，会生成 `build/compile_commands.json`，供 clangd 读取。

VS Code 打开仓库后，安装工作区推荐的 clangd 和 CMake Tools 扩展，并确保 `clangd` 在 `PATH` 中可用。工作区配置会在打开项目时运行 CMake，并让 clangd 使用 `build/` 下的编译数据库。也可以手动运行上面的配置命令，或执行 `CMake: Configure`；首次生成数据库后若编辑器诊断未刷新，可执行 `clangd: Restart language server`。已安装 Microsoft C/C++ 扩展时，工作区会关闭它的 IntelliSense，避免与 clangd 重复诊断。

`loom-plan-demo` 将乱序存放的四个节点准备为 `40 10 20 5`：生产者 40 必须先执行，两个就绪分支按 10、20 排列，最后汇合到 5。它说明节点 id 不等于执行次序。

`loom-contract-demo` 准备 FP32 → 显式 Cast → FP16 → 声明 FP32 累加的 KernelUnit 调用链。输出 slot 存储大小、符号名、数值契约及 workspace 要求；它不会实际进行类型转换或计算。

测试覆盖 Graph 创建/修改/删除、非法编辑拒绝、编辑器在容器扩容后的有效性，以及图级输入合法性、输入顺序与权重区分、DAG 结构、尺寸与跨度溢出、标量/空 tensor、padding/转置、非法 stride、入口重复/缺失、参数/端口数量、端口顺序、dtype/shape/layout/stride/对齐、数值契约以及快照所有权。使用实际检查而非依赖 `assert`，Release 构建也执行断言。

## 4 接下来的代码顺序

1. 实现实验 C ABI、CPU KernelUnit 动态加载与同步执行，跑通外部编译的两个小算子和 buffer 生命周期。
2. 接入 JSON schema、包加载与明确的版本/能力拒绝规则，再增加内存分配；不要在原型里另造 JSON 解析器。
3. 增加真实设备后端、显式 FP32/FP16 转换、混合累加契约及异步所有权验证，然后评审 ABI 冻结。
4. session API 稳定到可调用后接入可选 Python 绑定；自动精度搜索、量化和高级调度单独扩展。

每一步先完成可验证的调用闭环，再增加范围。首批原型不等于 M0 全部完成。
