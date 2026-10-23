#ifndef LOOM_PLAN_GRAPH_H
#define LOOM_PLAN_GRAPH_H

#include "plan/entry.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace loom::plan {

// S3: internal prototype API, not the published KernelUnit or host ABI.
using NodeId = std::uint32_t;
using SlotId = std::uint32_t;

// Slot 是执行图中一个 tensor 的逻辑位置, Slot 本身不是实际内存
// 将来执行时，runtime 才会为它分配或绑定 buffer
// 不同 slot 如果生命周期不重叠，还可以复用同一块内存
// 每个非 external slot 只由一个节点产生，但可以被多个节点读取
struct Slot {
  SlotId id;
  bool external = false;  // 是否由图外提供，比如输入和权重; graph nodes cannot write it.
  TensorDesc tensor;      // Required by prepare_program, ignored by prepare.
};

struct Node {
  NodeId id;
  std::vector<SlotId> inputs;
  std::vector<SlotId> outputs;
  std::vector<NodeId> after; // 数据流可以通过 slot, after 用来补充数据流没有表达的顺序约束
  EntryBinding binding;  // 使用哪个实现、哪份数值契约、什么参数, Required by prepare_program, ignored by prepare.
};

class NodeEditor;

struct Graph {
  std::vector<Slot> slots;
  std::vector<Node> nodes;
  // 每次调用按此顺序绑定的输入；必须是已有且不重复的 external slot。
  // 不在此列表中的 external slot 由加载阶段等其他途径提供，如权重/常量。
  std::vector<SlotId> inputs;
  std::vector<SlotId> outputs;

  // S3 建图/编辑 API。保留原始字段供序列化、检查和非法图测试使用。
  // ID 由调用方显式指定；add_input 同时创建 external slot 并追加图输入。
  // 缺失引用抛 out_of_range，重复 ID 等非法编辑抛 invalid_argument。
  // 全图合法性仍由 prepare/prepare_program 校验；链式调用不是整体事务。
  SlotId add_slot(SlotId id, TensorDesc tensor = {});
  SlotId add_external(SlotId id, TensorDesc tensor = {});
  SlotId add_input(SlotId id, TensorDesc tensor = {});
  NodeEditor add_node(NodeId id);
  NodeEditor node(NodeId id);

  void set_tensor(SlotId id, TensorDesc tensor);
  void set_inputs(std::vector<SlotId> ids);
  void set_outputs(std::vector<SlotId> ids);

  // 不级联删除或重连：slot 仍被节点/图接口引用，或 node 仍被 after 引用时拒绝。
  // 删除生产者会保留其 slot；后续需补生产者或清理 slot，再调用 prepare。
  void remove_slot(SlotId id);
  void remove_node(NodeId id);
};

// 按 ID 查找节点的轻量编辑器，避免保存 vector 元素引用后被扩容失效。
// Graph 必须存活且未被移动；删掉节点后使用编辑器会报错，重建同 ID 后指向新节点。
// reads/writes/after/bind 均替换对应字段；返回编辑器副本以便安全链式调用。
class NodeEditor {
 public:
  NodeEditor reads(std::vector<SlotId> ids) const;
  NodeEditor writes(std::vector<SlotId> ids) const;
  NodeEditor after(std::vector<NodeId> ids) const;
  NodeEditor bind(EntryBinding binding) const;

 private:
  friend struct Graph;
  NodeEditor(Graph& graph, NodeId id) : graph_(&graph), id_(id) {}
  Node& get() const;
  Graph* graph_;
  NodeId id_;
};

enum class ErrorCode {
  duplicate_node,
  duplicate_slot,
  unknown_slot,
  unknown_node,
  invalid_graph_input,
  duplicate_graph_input,
  missing_producer,
  multiple_producers,
  external_write,
  self_dependency,
  cycle,
  invalid_schedule,
  invalid_tensor,
  invalid_entry,
  duplicate_entry,
  unknown_entry,
  contract_mismatch,
};

struct Diagnostic {
  ErrorCode code;
  std::string message;
  std::optional<NodeId> node;
  std::optional<SlotId> slot;
  std::string field;
  std::string expected;
  std::string actual;
};

struct PrepareResult {
  std::vector<NodeId> order;  // Always empty on failure.
  std::optional<Diagnostic> error;
  bool ok() const { return !error.has_value(); }
};

// 校验图的输入接口、单生产者数据依赖和 after 约束，生成或检查提交顺序。
// 未指定顺序时，每次选择就绪节点中 id 最小者；显式顺序必须恰好覆盖全部节点
// 并满足依赖（显式空顺序不等于未指定）。失败时 order 为空，并返回诊断。
// 这里只检查图结构；tensor/入口契约由 prepare_program 检查，不执行 kernel。
PrepareResult prepare(
    const Graph& graph,
    const std::optional<std::vector<NodeId>>& explicit_order = std::nullopt);

}  // namespace loom::plan

#endif
