# 决策 0007 厂商库是 KernelUnit 的一等公民

- 状态：已接受
- 时间：2026-10

## 背景

公开数据显示，Agent 生成的 kernel 对上调优库普遍低于 1 倍，见 14 文档第 1 节。如果 ABI 只接受自写 kernel，本项目就必须在 cuDNN 最强的领域正面竞争，胜算很低。

## 决定

ABI 与包格式支持 `kind: vendor` 的 KernelUnit，把 cuDNN、cuBLAS、CUTLASS 实例化产物包装成普通 KernelUnit：

```json
{"id": "cudnn_v9", "kind": "vendor", "vendor": "cudnn", "min_arch": 80,
 "src": "src/cudnn_wrap.cpp", "params": {"algo": 0, "math_type": 1}}
```

引擎不得对厂商 KernelUnit 有任何特殊分支。它就是普通 KernelUnit，只是内部调用了库。

同时确立 01 文档 P16：Agent 的首要职责是选择，即判断哪里用库、哪里自写，其次才是自己写。

## 理由

第一，风险。不这么做，项目成败就绑在"Agent 超过 NVIDIA"这个不现实的赌注上。

第二，收益。10 文档第 4 节显示，三十个 KernelUnit 里只有八到十个需要自写，工作量降到三分之一。

第三，定位。从"要打败 cuDNN"变成"要解决 cuDNN 解决不了的融合问题"，后者可信得多。

## 后果

正面效果是第一阶段工作量降低，性能下限有保障，定位更可信。

负面效果有三点：需要处理库的 handle 生命周期，它是 per-device 的且有引用计数，且不能放进 `LmKernelArgs`，因为 ABI 需要稳定；库的 algo 有严格形状限制，必须声明适用域；引入第三方库的许可与分发问题，例如 cuDNN 不可再分发，需要目标机自行安装。

## 待补充

库 handle 的创建与销毁约定应写入 ABI 文档的附录，计划在 v1.1。另外需要考虑是否需要厂商 KernelUnit 的静态注册表，避免每台机器都依赖 cuDNN。

## 相关文档

06 文档第 7 节、01 文档 P16、12 文档质疑 2。
