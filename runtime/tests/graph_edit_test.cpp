#include "plan/program.h"

#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>

using namespace loom::plan;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Exception, class Action>
void rejects(Action action) {
  try {
    action();
  } catch (const Exception& error) {
    require(error.what()[0] != '\0', "empty edit diagnostic");
    return;
  }
  throw std::runtime_error("invalid edit was accepted");
}

const std::map<std::string, std::function<void()>> cases{
    {"create", [] {
       Graph graph;
       const auto input = graph.add_input(9);
       const auto weight = graph.add_external(2);
       const auto hidden = graph.add_slot(5);
       const auto output = graph.add_slot(7);
       graph.add_node(1).reads({hidden, weight}).writes({output});
       graph.add_node(40).reads({input}).writes({hidden});
       graph.set_outputs({output});
       const auto result = prepare(graph);
       require(result.ok() && result.order == std::vector<NodeId>({40, 1}),
               "builder lost data dependency");
       require(graph.inputs == std::vector<SlotId>({9}) &&
                   graph.outputs == std::vector<SlotId>({7}), "incorrect graph interface");
       require(graph.slots[0].external && graph.slots[1].external &&
                   !graph.slots[2].external, "incorrect slot roles");
     }},
    {"modify", [] {
       Graph graph;
       graph.add_input(9);
       graph.add_input(2);
       graph.add_slot(3);
       graph.add_slot(4);
       graph.add_node(1).reads({9}).writes({3});
       graph.add_node(8);
       graph.node(1).reads({2, 2}).writes({4}).after({8});
       graph.remove_slot(3);
       graph.set_inputs({2, 9});
       graph.set_outputs({4});
       TensorDesc tensor;
       tensor.dtype = DType::f32;
       tensor.shape = {6};
       graph.set_tensor(4, tensor);
       EntryBinding binding;
       binding.ref = {"unit", "impl"};
       binding.numerical_contract = "v1";
       binding.params = {1, 2};
       graph.node(1).bind(binding);
       const auto result = prepare(graph);
       require(result.ok() && result.order == std::vector<NodeId>({8, 1}),
               "edited after dependency lost");
       require(graph.nodes[0].inputs == std::vector<SlotId>({2, 2}) &&
                   graph.nodes[0].outputs == std::vector<SlotId>({4}),
               "reads/writes must replace rather than append");
       require(graph.inputs == std::vector<SlotId>({2, 9}) &&
                   graph.slots.back().tensor.shape == std::vector<std::int64_t>({6}) &&
                   graph.nodes[0].binding.params == std::vector<std::uint8_t>({1, 2}),
               "metadata edits lost");
     }},
    {"invalid_edits", [] {
       Graph graph;
       graph.add_input(0);
       graph.add_slot(1);
       graph.add_node(2).reads({0}).writes({1});
       graph.set_outputs({1});
       rejects<std::invalid_argument>([&] { graph.add_input(0); });
       rejects<std::invalid_argument>([&] { graph.add_input(1); });
       rejects<std::invalid_argument>([&] { graph.add_slot(0); });
       rejects<std::invalid_argument>([&] { graph.add_external(1); });
       rejects<std::invalid_argument>([&] { graph.add_node(2); });
       rejects<std::out_of_range>([&] { graph.node(99); });
       rejects<std::out_of_range>([&] { graph.set_tensor(99, {}); });
       rejects<std::out_of_range>([&] { graph.set_inputs({0, 99}); });
       rejects<std::invalid_argument>([&] { graph.set_inputs({1}); });
       rejects<std::invalid_argument>([&] { graph.set_inputs({0, 0}); });
       rejects<std::out_of_range>([&] { graph.set_outputs({0, 99}); });
       rejects<std::out_of_range>([&] { graph.node(2).reads({0, 99}); });
       rejects<std::out_of_range>([&] { graph.node(2).writes({1, 99}); });
       rejects<std::invalid_argument>([&] { graph.node(2).writes({0}); });
       rejects<std::invalid_argument>([&] { graph.node(2).writes({1, 1}); });
       rejects<std::out_of_range>([&] { graph.node(2).after({99}); });
       rejects<std::invalid_argument>([&] { graph.node(2).after({2}); });
       require(graph.slots.size() == 2 && graph.nodes.size() == 1 &&
                   graph.inputs == std::vector<SlotId>({0}) &&
                   graph.outputs == std::vector<SlotId>({1}) &&
                   graph.nodes[0].inputs == std::vector<SlotId>({0}) &&
                   graph.nodes[0].outputs == std::vector<SlotId>({1}) &&
                   graph.nodes[0].after.empty() && prepare(graph).ok(),
               "rejected edit changed the graph");
     }},
    {"editor_lifetime", [] {
       Graph graph;
       graph.add_input(0);
       auto editor = graph.add_node(1);
       for (NodeId id = 2; id < 300; ++id) graph.add_node(id);
       editor.reads({0}).after({299});  // Must survive vector reallocation.
       require(graph.nodes.front().inputs == std::vector<SlotId>({0}) &&
                   graph.nodes.front().after == std::vector<NodeId>({299}),
               "editor no longer refers to its id");
       graph.remove_node(1);
       rejects<std::out_of_range>([&] { editor.reads({0}); });
       graph.add_node(1);
       editor.reads({0});  // Editors select an id, not a historical node instance.
       require(graph.nodes.back().inputs == std::vector<SlotId>({0}),
               "editor did not select recreated id");
     }},
    {"remove", [] {
       Graph graph;
       graph.add_input(0);
       graph.add_slot(1);
       graph.add_node(2).reads({0}).writes({1});
       graph.add_node(3).after({2});
       graph.set_outputs({1});
       rejects<std::invalid_argument>([&] { graph.remove_slot(0); });
       rejects<std::invalid_argument>([&] { graph.remove_slot(1); });
       rejects<std::invalid_argument>([&] { graph.remove_node(2); });
       graph.set_inputs({});
       graph.set_outputs({});
       rejects<std::invalid_argument>([&] { graph.remove_slot(0); });
       rejects<std::invalid_argument>([&] { graph.remove_slot(1); });
       graph.node(3).after({});
       graph.remove_node(2);
       const auto incomplete = prepare(graph);
       require(!incomplete.ok() && incomplete.error->code == ErrorCode::missing_producer,
               "removing a producer must not silently remove its tensor");
       graph.remove_slot(0);
       graph.remove_slot(1);
       graph.remove_node(3);
       require(graph.nodes.empty() && graph.slots.empty() && prepare(graph).ok(),
               "explicit graph cleanup failed");
       rejects<std::out_of_range>([&] { graph.remove_node(99); });
       rejects<std::out_of_range>([&] { graph.remove_slot(99); });
     }},
    {"global_validation", [] {
       Graph graph;
       graph.add_slot(1);
       graph.add_node(2).writes({1});
       graph.add_node(3).writes({1});
       auto result = prepare(graph);
       require(!result.ok() && result.error->code == ErrorCode::multiple_producers,
               "global writer validation bypassed");
       graph.node(3).writes({}).after({2});
       graph.node(2).after({3});
       result = prepare(graph);
       require(!result.ok() && result.error->code == ErrorCode::cycle,
               "global cycle validation bypassed");
     }},
    {"program", [] {
       TensorDesc tensor;
       tensor.dtype = DType::f32;
       tensor.shape = {2, 3};
       EntryContract entry;
       entry.ref = {"identity", "cpu"};
       entry.symbol = "test_identity";
       entry.inputs = {{"x", tensor}};
       entry.outputs = {{"y", tensor}};
       entry.numerical_contract = "exact/v1";
       EntryBinding binding;
       binding.ref = entry.ref;
       binding.numerical_contract = "exact/v1";
       Graph graph;
       const auto input = graph.add_input(0, tensor);
       const auto output = graph.add_slot(1, tensor);
       graph.add_node(10).reads({input}).writes({output}).bind(binding);
       graph.set_outputs({output});
       const auto result = prepare_program(graph, {entry});
       require(result.ok() && result.value->tensors.at(1).info.span_bytes == 24 &&
                   result.value->calls[0].entry.symbol == "test_identity",
               "edited graph failed full contract validation");
       binding.numerical_contract = "different/v1";
       graph.node(10).bind(binding);
       require(!prepare_program(graph, {entry}).ok() &&
                   result.value->calls[0].entry.numerical_contract == "exact/v1",
               "editing graph changed an existing prepared snapshot");
     }},
};

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 || cases.count(argv[1]) == 0) return 2;
  try {
    cases.at(argv[1])();
  } catch (const std::exception& error) {
    std::cerr << argv[1] << ": " << error.what() << '\n';
    return 1;
  }
}
