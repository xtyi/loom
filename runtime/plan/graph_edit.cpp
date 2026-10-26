#include "plan/graph.h"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

namespace loom::plan {
namespace {

template <class Id>
bool contains(const std::vector<Id>& ids, Id id) {
  return std::find(ids.begin(), ids.end(), id) != ids.end();
}

Slot& find_slot(Graph& graph, SlotId id) {
  const auto it = std::find_if(graph.slots.begin(), graph.slots.end(),
                               [id](const Slot& slot) { return slot.id == id; });
  if (it == graph.slots.end()) {
    throw std::out_of_range("unknown slot " + std::to_string(id));
  }
  return *it;
}

Node& find_node(Graph& graph, NodeId id) {
  const auto it = std::find_if(graph.nodes.begin(), graph.nodes.end(),
                               [id](const Node& node) { return node.id == id; });
  if (it == graph.nodes.end()) {
    throw std::out_of_range("unknown node " + std::to_string(id));
  }
  return *it;
}

SlotId next_slot_id(const Graph& graph) {
  SlotId next = 0;
  for (const auto& slot : graph.slots) {
    if (slot.id == std::numeric_limits<SlotId>::max()) {
      throw std::overflow_error("cannot allocate beyond maximum slot id");
    }
    next = std::max(next, slot.id + 1);
  }
  return next;
}

SlotId append_slot(Graph& graph, SlotId id, bool external, TensorDesc tensor) {
  graph.slots.push_back({id, external, std::move(tensor)});
  return id;
}

}  // namespace

SlotId Graph::add_slot(TensorDesc tensor) {
  return append_slot(*this, next_slot_id(*this), false, std::move(tensor));
}

SlotId Graph::add_external(TensorDesc tensor) {
  return append_slot(*this, next_slot_id(*this), true, std::move(tensor));
}

SlotId Graph::add_input(TensorDesc tensor) {
  const auto id = next_slot_id(*this);
  // Also detect an existing input id if raw-field edits made the graph invalid.
  if (contains(inputs, id)) {
    throw std::invalid_argument("duplicate graph input " + std::to_string(id));
  }
  auto new_inputs = inputs;
  new_inputs.push_back(id);
  append_slot(*this, id, true, std::move(tensor));
  inputs.swap(new_inputs);
  return id;
}

NodeEditor Graph::add_node(NodeId id) {
  for (const auto& existing : nodes) {
    if (existing.id == id) {
      throw std::invalid_argument("duplicate node " + std::to_string(id));
    }
  }
  Node created{};
  created.id = id;
  nodes.push_back(std::move(created));
  return NodeEditor(*this, id);
}

NodeEditor Graph::node(NodeId id) {
  find_node(*this, id);
  return NodeEditor(*this, id);
}

void Graph::set_tensor(SlotId id, TensorDesc tensor) {
  find_slot(*this, id).tensor = std::move(tensor);
}

void Graph::set_inputs(std::vector<SlotId> ids) {
  std::set<SlotId> seen;
  for (const auto id : ids) {
    if (!find_slot(*this, id).external) {
      throw std::invalid_argument("graph input must be external: " + std::to_string(id));
    }
    if (!seen.insert(id).second) {
      throw std::invalid_argument("duplicate graph input " + std::to_string(id));
    }
  }
  inputs = std::move(ids);
}

void Graph::set_outputs(std::vector<SlotId> ids) {
  for (const auto id : ids) find_slot(*this, id);
  outputs = std::move(ids);
}

void Graph::remove_slot(SlotId id) {
  find_slot(*this, id);
  if (contains(inputs, id) || contains(outputs, id)) {
    throw std::invalid_argument("slot is referenced by graph interface: " +
                                std::to_string(id));
  }
  for (const auto& existing : nodes) {
    if (contains(existing.inputs, id) || contains(existing.outputs, id)) {
      throw std::invalid_argument("slot is referenced by node " +
                                  std::to_string(existing.id));
    }
  }
  slots.erase(std::remove_if(slots.begin(), slots.end(),
                             [id](const Slot& slot) { return slot.id == id; }),
              slots.end());
}

void Graph::remove_node(NodeId id) {
  find_node(*this, id);
  for (const auto& existing : nodes) {
    if (existing.id != id && contains(existing.after, id)) {
      throw std::invalid_argument("node is referenced by after of node " +
                                  std::to_string(existing.id));
    }
  }
  nodes.erase(std::remove_if(nodes.begin(), nodes.end(),
                             [id](const Node& node) { return node.id == id; }),
              nodes.end());
}

Node& NodeEditor::get() const { return find_node(*graph_, id_); }

NodeEditor NodeEditor::reads(std::vector<SlotId> ids) const {
  auto& selected = get();
  for (const auto id : ids) find_slot(*graph_, id);
  selected.inputs = std::move(ids);
  return *this;
}

NodeEditor NodeEditor::writes(std::vector<SlotId> ids) const {
  auto& selected = get();
  std::set<SlotId> seen;
  for (const auto id : ids) {
    if (find_slot(*graph_, id).external) {
      throw std::invalid_argument("cannot write external slot " + std::to_string(id));
    }
    if (!seen.insert(id).second) {
      throw std::invalid_argument("duplicate output slot " + std::to_string(id));
    }
  }
  selected.outputs = std::move(ids);
  return *this;
}

NodeEditor NodeEditor::after(std::vector<NodeId> ids) const {
  auto& selected = get();
  for (const auto id : ids) {
    find_node(*graph_, id);
    if (id == id_) throw std::invalid_argument("node cannot depend on itself");
  }
  selected.after = std::move(ids);
  return *this;
}

NodeEditor NodeEditor::bind(EntryBinding binding) const {
  get().binding = std::move(binding);
  return *this;
}

}  // namespace loom::plan
