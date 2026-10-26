# 独立参考与包验收工具

`scripts/onnx_workbench.py` 提供下载、原始模型参考执行和当前 CLI 数值检查。它不生成模型 kernel 或 plan。当前范围是权重内嵌的单文件 ONNX、单输入、静态 profile、紧凑 FP32 输入输出；不匹配时使用适合该模型的独立验证程序。

## 环境与参考

变量设为实际绝对路径，`SKILL` 为本技能目录，`WORK` 为本次模型工作目录。依赖装在隔离环境，不改 runtime 的 C++ 构建依赖：

```sh
python3 -m venv "$WORK/venv"
"$WORK/venv/bin/python" -m pip install numpy onnx onnxruntime
"$WORK/venv/bin/python" "$SKILL/scripts/onnx_workbench.py" fetch \
  --out "$WORK/resnet18-v1-7.onnx"
"$WORK/venv/bin/python" "$SKILL/scripts/onnx_workbench.py" reference \
  --model "$WORK/resnet18-v1-7.onnx" --shape 1,3,224,224 \
  --out "$WORK/reference"
```

下载需要 `curl`；锁定 revision、大小和 SHA-256，下载失败不发布半成品。`reference` 运行 `onnx.checker` 和独立 ORT CPU 会话，关闭图优化、固定线程数，生成：

- `model-info.json`：模型哈希、IR、opset、节点/initializer 和输入输出信息。属性文本仅供分析，生成器仍需读原始 ONNX。
- `reference.json`：实际输入输出顺序、shape、版本、字节序及每份数据的 SHA-256。
- `cases/`：zeros、ones、固定种子 normal、ramp，以及相应完整参考输出。可重复 `--input /path/input.bin` 加入真实图片预处理后的张量。

参考目录必须是新目录，避免覆盖既有语料。实际构建时保存依赖锁，例如 `python -m pip freeze`，并记录编译器和 flags。相同随机种子不代替数据文件哈希。工具使用系统字节序，跨字节序环境应重新生成参考。

## 执行候选包并比较

```sh
"$WORK/venv/bin/python" "$SKILL/scripts/onnx_workbench.py" check \
  --model "$WORK/resnet18-v1-7.onnx" \
  --reference "$WORK/reference" \
  --package "$WORK/dist/resnet18.loom" \
  --cli "$REPO/build-model/loom-cli" \
  --atol 0.0001 --rtol 0.0001 \
  --report "$WORK/validation.json"
```

工具先核对原模型哈希、语料文件哈希、包的有序输入输出 dtype/shape/stride，再执行 `validate` 和逐例 `run`；每例使用新临时输出目录。逐元素要求：

```text
abs(candidate - reference) <= atol + rtol * abs(reference)
```

报告记录实际执行结果、最大绝对/相对误差、最大绝对误差位置和超差元素数。输入、参考或候选的 NaN/Inf 都拒绝。分类 Top-1 一致度只是附加项。进程失败、超时、文件大小不符、哈希不符、数值超差均返回非零并写失败报告。

默认 `atol=rtol=1e-4` 是此 FP32 实验的初始验收预算，不是所有模型的通用保证。预算在调优前固定；需要调整时解释误差来源和应用要求，重新明确预算，不可修改参考数据来配合实现。

通过此工具只证明测试输入上的数值一致性。还需按 SKILL.md 做可重建性、迁移运行和实际依赖检查；数据集准确率、性能、低精度版本和新 shape 各有单独验证范围。
