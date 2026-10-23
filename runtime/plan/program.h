#ifndef LOOM_PLAN_PROGRAM_H
#define LOOM_PLAN_PROGRAM_H

#include "plan/graph.h"

#include <map>

namespace loom::plan {

struct PreparedCall {
  NodeId node; // 原图中的节点 ID，用于日志、报错和定位
  EntryContract entry;  // Owned snapshot, not a reference to mutable registry data.
  std::vector<SlotId> inputs; // 输入 slot ID，顺序对应入口的输入端口
  std::vector<SlotId> outputs; // 输出 slot ID，顺序对应入口的输出端口
  std::vector<std::uint8_t> params; // 传给 kernel 的不透明参数字节，runtime 已检查大小，暂不解释内容
};

struct PreparedTensor {
  bool external;
  TensorInfo info;
};

struct PreparedProgram {
  std::map<SlotId, PreparedTensor> tensors;
  std::vector<PreparedCall> calls;  // In validated submission order.
  std::vector<SlotId> inputs;  // 保留调用方输入顺序，不包括单独加载的权重/常量。
  std::vector<SlotId> outputs;
};

struct ProgramResult {
  std::optional<PreparedProgram> value;  // Absent on every failure.
  std::optional<Diagnostic> error;
  bool ok() const { return value.has_value(); }
};

// Validates topology, tensors, registry entries and every positional binding.
// All registry entries are validated, including unused entries. No memory is
// allocated for tensors/workspace; no symbol is resolved or invoked. A successful
// result is an owned metadata snapshot, not proof of a kernel's actual behavior.
ProgramResult prepare_program(
    const Graph& graph, const std::vector<EntryContract>& entries,
    const std::optional<std::vector<NodeId>>& explicit_order = std::nullopt);

}  // namespace loom::plan

#endif
