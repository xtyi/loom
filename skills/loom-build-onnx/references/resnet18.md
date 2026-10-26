# 默认模型：ResNet-18 v1 / ONNX / FP32

选择理由：常见的 ImageNet 分类网络，普通卷积和残差块较容易建立可诊断基线，同时覆盖跨层依赖、池化与分类头。第一轮目标是模型计算与目录包链路跑通，性能调优随后进行。

来源为 [ONNX Model Zoo 的迁移仓库](https://huggingface.co/onnxmodelzoo/resnet18-v1-7)，具体文件和哈希锁在同目录 `resnet18.lock.json`。这是模型卡所述的 **MXNet 导出的 ResNet-v1**，不要声称与 torchvision 的任意一组 ResNet-18 权重完全相同。模型卡标注 Apache-2.0；保留来源和适用许可。

ONNX Model Zoo [原仓库公告](https://github.com/onnx/models)已说明旧 Git LFS 下载路径停止服务，优先使用上述锁定的 Hugging Face URL。不要把 Git LFS pointer 当作 ONNX 文件。

默认 profile：单输入、batch=1、RGB NCHW、FP32 `[1,3,224,224]`，分类输出 `[1,1000]`。实际输入/输出名字、节点数、属性、IR 和 opset 必须从文件读取；文件名不能代替检查。原模型中的符号 batch 只在此 profile 固定为 1，不代表 Loom 支持动态 batch。

对上述锁定文件实测：IR=3，默认域 **opset=8**，输入 `data` 的 shape 为 `[N,3,224,224]`，输出 `resnetv15_dense0_fwd` 为 `[N,1000]`。文件名及模型卡表格中的 `7` 与实际 opset 不同，实现时按文件中的 8 解释。共 69 个 node、102 个 initializer：Conv 20、BatchNormalization 20、Relu 17、Add 8，以及 MaxPool、GlobalAveragePool、Flatten、Gemm 各 1 个。四组合成输入已用 ORT CPU 执行参考端；这不表示 Loom 已经具备对应 kernel。

真实图片预处理按[模型卡](https://huggingface.co/onnxmodelzoo/resnet18-v1-7/blob/e7cb849a949bdceba02356b8b923d53cc01108e1/README.md)：RGB；保持比例将短边缩放到 256；中心裁剪 224；转为 FP32；按 `(pixel/255 - mean)/std` 归一化，mean 为 `[0.485,0.456,0.406]`，std 为 `[0.229,0.224,0.225]`；HWC 转 NCHW 并增加 batch 维。固定所用图像库、插值和取整规则，参考端与候选端使用同一已预处理张量。

输出按实际图的定义比较，不要自行添加 Softmax。展示分类概率时可在推理输出之外后处理，但这不能替代原始输出比较。类别文本映射另记录来源和索引约定。

实现时特别检查：

- Conv 的 OIHW 权重、strides、pads、dilations、group 和 bias 是否存在。
- BatchNormalization 的推理公式、epsilon 和旧 opset 属性；若折叠，记录 `W'=W*gamma/sqrt(var+eps)`、`b'=beta+(b-mean)*gamma/sqrt(var+eps)`，缺省卷积 bias 取 0，并验证误差。
- MaxPool 的 padding、边界和输出取整规则；不要凭现代框架默认值推测旧 opset。
- 残差分支的实际 Add 输入与广播规则；GlobalAveragePool 的归约维度；Flatten 的 axis；Gemm 的 alpha/beta/transA/transB。

上面是检查点，不是该导出图的算子列表。以 `reference` 生成的 `model-info.json` 和原始 ONNX 为准。需要查语义时使用 [ONNX 官方算子文档及版本历史](https://onnx.ai/onnx/operators/)，针对实际 opset 阅读。
