# 当前 Loom 接口快照

此参考针对实验 CPU 路径，使用前读取仓库当前源码。草案文档中的动态 shape、arena、alias、variant、GPU、`lm_ku_*_v1` 都不是当前已实现能力。

## 事实来源与编译

- `runtime/abi/experimental.h`：结构体、数据类型、导出函数与指针生命周期。
- `schemas/experimental/{plan,kernel,package}.schema.json`、`docs/17-json-package.md`：JSON 格式。
- `runtime/plan/{entry,tensor,graph}.h`：元数据契约。
- `runtime/cpu/`、`runtime/io/`、`runtime/cli/main.cpp`：加载和执行行为。
- `examples/kernels/`、`examples/packages/cpu/`、`cmake/ExamplePackage.cmake`：配套可执行范例。

```sh
cmake -S "$REPO" -B "$REPO/build-model" \
  -DCMAKE_BUILD_TYPE=Release -DLOOM_ENABLE_EXPERIMENTAL_ABI=ON
cmake --build "$REPO/build-model" -j4
"$REPO/build-model/loom-cli" inspect "$PACKAGE"
"$REPO/build-model/loom-cli" validate "$PACKAGE"
"$REPO/build-model/loom-cli" run "$PACKAGE" "$FRESH_OUTPUT_DIR" "$INPUT_BIN"
```

开启实验 ABI 后才有 CPU `run`；`inspect/validate` 不加载动态库，不能发现二进制 caps 错误或数值错误。当前 Session 对每次运行分配独立中间存储，无 arena 复用；估算内存时不能假定生命周期重用。

## 目录包和 JSON

```text
resnet18.loom/
  package.json
  plans/main.json
  kernels/<unit>/ku.json
  kernels/<unit>/lib<unit>.so
  weights/<tensor>.bin
```

以 Schema/可执行示例为模板，注意这些容易生成错的规则：

| 项目 | 当前规则 |
|---|---|
| format | 分别为 `loom-package-exp/1`、`loom-plan-exp/1`、`loom-kernel-exp/1`，精确匹配 |
| target | `backend=cpu`、`abi=loom.cpu.exp.v1`，os/arch/endianness/pointer_bits 必须匹配宿主；从构建环境派生 |
| 路径 | plan、manifest、library、constant 路径全部相对**包根**，不是相对 manifest 目录；禁止绝对路径和 `..` |
| 扩展信息 | 执行字段中未知 key 会拒绝。自定义来源/预处理等信息放根级 `meta` 或包外报告 |
| ID | JSON 中的 slot/node ID 显式保存；唯一、引用有效，不必连续；C++ Graph API 新建 slot 则自动分配 ID |
| 张量 | rank 0..8、静态 shape；`dtype` 必须显式；省略 strides 或 `[]` 为紧凑行主序；stride 单位为元素 |
| layout | 不解释的整数标记，默认 0；不是字符串 `NCHW`，不自动转置 |
| 对齐 | slot 的保证至少满足入口要求；2 的幂；动态库 caps 的 min_buffer_alignment 不得超过各端口声明 |
| 节点 | `inputs/outputs` 按位置绑定；`binding` 指明 kernel、implementation、numerical_contract 和可选 params_hex |
| 入口 | 一个 `(kernel,implementation)` 对应唯一 EntryContract；节点和 manifest 张量描述必须匹配 |
| 常量 | 所有 external slot 中除 `plan.inputs` 外的部分由 package.constants 完整且唯一覆盖 |
| 权重文件 | 原生 dtype、宿主字节序、精确存储跨度的原始字节；无 `.npy` 头；有 padding 时也必须写入 |
| schedule | 可省略；若提供，`order` 必须恰好覆盖所有 node 且满足依赖；分支应正确连线，不靠执行顺序掩盖缺边 |

`package.json` 必填 `format/name/target/plan/kernels/constants`。kernel manifest 必填 `format/kernel/implementation/symbol/inputs/outputs/numerical_contract/library`；参数和 workspace 按需声明。完整字段使用源码 Schema，不在此另造格式。

## ABI 与资源

编译时包含仓库 `runtime/abi/experimental.h`，不复制一份自己猜测的结构定义。C++ 实现使用 C linkage。动态库导出：

```c
const LmExpCaps* lm_ku_caps_exp_v1(void);
int32_t lm_ku_entry_exp_v1(const LmExpKernelArgs* args, void* stream);
```

- CPU 同步，`stream == NULL`，所有工作结束后返回；成功为 0，kernel 错误为 1..255；不得让异常穿过 ABI。
- 只在调用期间借用输入、输出、params 和 workspace。输入和 params 只读；output descriptor 不可改，output data 可写。不得保留指针作为下次调用状态。
- `params` 无对齐保证，用 `memcpy` 解码；manifest.params_bytes、binding.params_hex 长度、caps.params_bytes 一致。
- workspace 是未初始化、可被后续调用复用的 scratch。声明固定 bytes/alignment，每次初始化需要读取的区域。不得把它视为持久权重缓存。
- caps 的版本、CPU backend、同步 flags、入口身份、数值契约 ID、端口数、参数和 workspace 声明必须匹配 manifest，相关字符串在卸载前有效。
- 当前不支持 alias/原地更新，external slot 只读；kernel 输出与输入使用独立 slot。不要把整个模型的中间数组全部放进巨型栈帧。

当前运行时对 `numerical_contract` 做身份匹配，既不解析累加精度也不执行数值检查。内容哈希、签名、隔离执行和细粒度 CPU ISA 检查也尚未实现；需要的构建依赖和 ISA 要由产物报告说明。

## CLI 文件协议

输入文件按 `plan.inputs` 的数组顺序传递，输出按 `plan.outputs` 的**位置**命名为 `0.bin`、`1.bin` 等，不是 slot ID。文件保存完整存储跨度，包含 stride padding。已有输出文件拒绝覆盖，每个测试使用新的输出目录。默认验证工具只支持紧凑、单输入、全 FP32 的 profile，遇到其他布局会显式拒绝。
