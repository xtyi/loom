#include "plan/graph.h"

#include <map>
#include <set>
#include <utility>

namespace loom::plan {
namespace {

PrepareResult fail(ErrorCode code, std::string message,
                   std::optional<NodeId> node = std::nullopt,
                   std::optional<SlotId> slot = std::nullopt) {
  return {{}, Diagnostic{code, std::move(message), node, slot, {}, {}, {}}};
}

struct SlotInfo {
  bool external;
  std::optional<NodeId> producer;
};

}  // namespace

// 准备图的合法提交顺序，不修改原图，也不分配 tensor 内存或调用 kernel。
// 1. 检查 slot/node id 唯一；图级 inputs 必须存在、不重复且为 external。
//    inputs 只列出本次调用传入的数据；权重等 external slot 可以不在其中。
// 2. 先收集所有输出的生产者，拒绝写 external 或一个 slot 有多个输出绑定；
//    再检查节点输入、图输出和 after 引用，确保所有非 external slot 有生产者。
// 3. 从 producer → consumer 自动建立数据依赖，再合并 after 额外依赖并去重；
//    拒绝读取自己的输出、自依赖及依赖环，不按节点编号猜测先后关系。
// 4. 用 Kahn 拓扑排序，每次取就绪集合中最小 id，得到确定性的默认顺序。
// 5. 若给了 explicit_order，检查它包含全部节点恰好一次，且满足每条依赖；
//    合法时使用该顺序，不擅自修正非法顺序。
// 成功返回完整 order；失败返回诊断且 order 为空，不泄漏半成品顺序。
// 本函数忽略 tensor 描述和入口 binding：结构正确不等于可以执行，完整的
// 静态契约检查应调用 prepare_program；实际内存/设备/数值行为还需后续验证。
PrepareResult prepare(
    const Graph& graph,
    const std::optional<std::vector<NodeId>>& explicit_order) {
  std::map<SlotId, SlotInfo> slots;
  std::map<NodeId, const Node*> nodes;
  for (const auto& slot : graph.slots) {
    if (!slots.emplace(slot.id, SlotInfo{slot.external, std::nullopt}).second) {
      return fail(ErrorCode::duplicate_slot, "duplicate slot id", {}, slot.id);
    }
  }
  for (const auto& node : graph.nodes) {
    if (!nodes.emplace(node.id, &node).second) {
      return fail(ErrorCode::duplicate_node, "duplicate node id", node.id);
    }
  }

  std::set<SlotId> graph_inputs;
  for (const auto input : graph.inputs) {
    const auto found = slots.find(input);
    if (found == slots.end()) {
      return fail(ErrorCode::unknown_slot, "unknown graph input", {}, input);
    }
    if (!graph_inputs.insert(input).second) {
      return fail(ErrorCode::duplicate_graph_input, "duplicate graph input", {}, input);
    }
    if (!found->second.external) {
      return fail(ErrorCode::invalid_graph_input,
                  "graph input must reference an external slot", {}, input);
    }
  }

  // Discover every producer before reading any input; id and file order carry
  // no execution semantics. Maps also give stable validation traversal.
  for (const auto& [id, node] : nodes) {
    for (const auto output : node->outputs) {
      auto found = slots.find(output);
      if (found == slots.end()) {
        return fail(ErrorCode::unknown_slot, "unknown output slot", id, output);
      }
      auto& slot = found->second;
      if (slot.external) {
        return fail(ErrorCode::external_write, "cannot write external slot", id,
                    output);
      }
      if (slot.producer) {
        return fail(ErrorCode::multiple_producers,
                    "slot must have exactly one output binding", id, output);
      }
      slot.producer = id;
    }
  }

  std::map<NodeId, std::set<NodeId>> successors;
  std::map<NodeId, std::size_t> indegree;
  for (const auto& [id, node] : nodes) {
    (void)node;
    indegree.emplace(id, 0);
  }
  const auto add_edge = [&](NodeId from, NodeId to) {
    if (successors[from].insert(to).second) {
      ++indegree.at(to);
    }
  };
  for (const auto& [id, node] : nodes) {
    for (const auto input : node->inputs) {
      const auto found = slots.find(input);
      if (found == slots.end()) {
        return fail(ErrorCode::unknown_slot, "unknown input slot", id, input);
      }
      const auto& slot = found->second;
      if (slot.external) {
        continue;
      }
      if (!slot.producer) {
        return fail(ErrorCode::missing_producer, "input has no producer", id,
                    input);
      }
      if (*slot.producer == id) {
        return fail(ErrorCode::self_dependency,
                    "node reads its own output; in-place updates unsupported",
                    id, input);
      }
      add_edge(*slot.producer, id);
    }
    for (const auto predecessor : node->after) {
      if (nodes.count(predecessor) == 0) {
        return fail(ErrorCode::unknown_node, "unknown after predecessor", id);
      }
      if (predecessor == id) {
        return fail(ErrorCode::self_dependency, "node depends on itself", id);
      }
      add_edge(predecessor, id);
    }
  }
  for (const auto output : graph.outputs) {
    if (slots.count(output) == 0) {
      return fail(ErrorCode::unknown_slot, "unknown graph output", {}, output);
    }
  }
  for (const auto& [id, slot] : slots) {
    if (!slot.external && !slot.producer) {
      return fail(ErrorCode::missing_producer, "slot has no producer", {}, id);
    }
  }

  std::set<NodeId> ready;
  for (const auto& [id, degree] : indegree) {
    if (degree == 0) {
      ready.insert(id);
    }
  }
  std::vector<NodeId> order;
  order.reserve(nodes.size());
  while (!ready.empty()) {
    const auto id = *ready.begin();
    ready.erase(ready.begin());
    order.push_back(id);
    const auto next = successors.find(id);
    if (next == successors.end()) {
      continue;
    }
    for (const auto successor : next->second) {
      if (--indegree.at(successor) == 0) {
        ready.insert(successor);
      }
    }
  }
  if (order.size() != nodes.size()) {
    for (const auto& [id, degree] : indegree) {
      if (degree != 0) {
        // This id is blocked by a cycle; it need not be on the cycle itself.
        return fail(ErrorCode::cycle, "cyclic dependencies block this node", id);
      }
    }
  }

  if (explicit_order) {
    if (explicit_order->size() != nodes.size()) {
      return fail(ErrorCode::invalid_schedule,
                  "schedule must contain every node exactly once");
    }
    std::map<NodeId, std::size_t> position;
    for (std::size_t i = 0; i < explicit_order->size(); ++i) {
      const auto id = (*explicit_order)[i];
      if (nodes.count(id) == 0) {
        return fail(ErrorCode::invalid_schedule, "unknown scheduled node", id);
      }
      if (!position.emplace(id, i).second) {
        return fail(ErrorCode::invalid_schedule, "duplicate scheduled node", id);
      }
    }
    for (const auto& [from, next] : successors) {
      for (const auto to : next) {
        if (position.at(from) >= position.at(to)) {
          return fail(ErrorCode::invalid_schedule,
                      "schedule places a node before its predecessor", to);
        }
      }
    }
    order = *explicit_order;
  }
  return {std::move(order), std::nullopt};
}

}  // namespace loom::plan
