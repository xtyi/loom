#include "plan/program.h"

#include <functional>
#include <iostream>
#include <limits>
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
       const auto input = graph.add_input();
       const auto weight = graph.add_external();
       const auto hidden = graph.add_slot();
       const auto output = graph.add_slot();
       graph.add_node(1).reads({hidden, weight}).writes({output});
       graph.add_node(40).reads({input}).writes({hidden});
       graph.set_outputs({output});
       const auto result = prepare(graph);
       require(result.ok() && result.order == std::vector<NodeId>({40, 1}),
               "builder lost data dependency");
       require(input == 0 && weight == 1 && hidden == 2 && output == 3,
               "slot creation APIs must share one ID sequence");
       require(graph.inputs == std::vector<SlotId>({input}) &&
                   graph.outputs == std::vector<SlotId>({output}), "incorrect graph interface");
       require(graph.slots[0].external && graph.slots[1].external &&
                   !graph.slots[2].external, "incorrect slot roles");
     }},
    {"slot_allocation", [] {
       Graph graph;
       // Imported/raw slots need not be dense or sorted.
       graph.slots = {{9, true}, {0, true}, {4, true}};
       graph.inputs = {4, 0};
       const auto hidden = graph.add_slot();
       const auto weight = graph.add_external();
       const auto input = graph.add_input();
       require(hidden == 10 && weight == 11 && input == 12 &&
                   graph.slots[3].id == hidden && graph.slots[4].id == weight &&
                   graph.slots[5].id == input &&
                   graph.inputs == std::vector<SlotId>({4, 0, input}),
               "allocation collided with imported slots or changed input order");
       graph.remove_slot(hidden);
       require(graph.add_slot() == 13, "removing a slot caused an ID collision");
     }},
    {"slot_id_overflow", [] {
       const auto max = std::numeric_limits<SlotId>::max();
       Graph graph;
       graph.slots.push_back({max - 1, true});
       require(graph.add_input() == max, "maximum slot ID must remain usable");
       rejects<std::overflow_error>([&] { graph.add_slot(); });
       rejects<std::overflow_error>([&] { graph.add_external(); });
       rejects<std::overflow_error>([&] { graph.add_input(); });
       require(graph.slots.size() == 2 && graph.slots.back().id == max &&
                   graph.inputs == std::vector<SlotId>({max}) && prepare(graph).ok(),
               "allocation overflow changed the graph");
     }},
    {"modify", [] {
       Graph graph;
       const auto first = graph.add_input();
       const auto second = graph.add_input();
       const auto old_output = graph.add_slot();
       const auto output = graph.add_slot();
       graph.add_node(1).reads({first}).writes({old_output});
       graph.add_node(8);
       graph.node(1).reads({second, second}).writes({output}).after({8});
       graph.remove_slot(old_output);
       graph.set_inputs({second, first});
       graph.set_outputs({output});
       TensorDesc tensor;
       tensor.dtype = DType::f32;
       tensor.shape = {6};
       graph.set_tensor(output, tensor);
       EntryBinding binding;
       binding.ref = {"unit", "impl"};
       binding.numerical_contract = "v1";
       binding.params = {1, 2};
       graph.node(1).bind(binding);
       const auto result = prepare(graph);
       require(result.ok() && result.order == std::vector<NodeId>({8, 1}),
               "edited after dependency lost");
       require(graph.nodes[0].inputs == std::vector<SlotId>({second, second}) &&
                   graph.nodes[0].outputs == std::vector<SlotId>({output}),
               "reads/writes must replace rather than append");
       require(graph.inputs == std::vector<SlotId>({second, first}) &&
                   graph.slots.back().tensor.shape == std::vector<std::int64_t>({6}) &&
                   graph.nodes[0].binding.params == std::vector<std::uint8_t>({1, 2}),
               "metadata edits lost");
     }},
    {"invalid_edits", [] {
       Graph graph;
       const auto input = graph.add_input();
       const auto output = graph.add_slot();
       graph.add_node(2).reads({input}).writes({output});
       graph.set_outputs({output});
       rejects<std::invalid_argument>([&] { graph.add_node(2); });
       rejects<std::out_of_range>([&] { graph.node(99); });
       rejects<std::out_of_range>([&] { graph.set_tensor(99, {}); });
       rejects<std::out_of_range>([&] { graph.set_inputs({input, 99}); });
       rejects<std::invalid_argument>([&] { graph.set_inputs({output}); });
       rejects<std::invalid_argument>([&] { graph.set_inputs({input, input}); });
       rejects<std::out_of_range>([&] { graph.set_outputs({input, 99}); });
       rejects<std::out_of_range>([&] { graph.node(2).reads({input, 99}); });
       rejects<std::out_of_range>([&] { graph.node(2).writes({output, 99}); });
       rejects<std::invalid_argument>([&] { graph.node(2).writes({input}); });
       rejects<std::invalid_argument>([&] { graph.node(2).writes({output, output}); });
       rejects<std::out_of_range>([&] { graph.node(2).after({99}); });
       rejects<std::invalid_argument>([&] { graph.node(2).after({2}); });
       require(graph.slots.size() == 2 && graph.nodes.size() == 1 &&
                   graph.inputs == std::vector<SlotId>({input}) &&
                   graph.outputs == std::vector<SlotId>({output}) &&
                   graph.nodes[0].inputs == std::vector<SlotId>({input}) &&
                   graph.nodes[0].outputs == std::vector<SlotId>({output}) &&
                   graph.nodes[0].after.empty() && prepare(graph).ok(),
               "rejected edit changed the graph");
       Graph invalid;
       invalid.inputs = {0};
       rejects<std::invalid_argument>([&] { invalid.add_input(); });
       require(invalid.slots.empty() && invalid.inputs == std::vector<SlotId>({0}),
               "rejected input allocation changed the graph");
     }},
    {"editor_lifetime", [] {
       Graph graph;
       const auto input = graph.add_input();
       auto editor = graph.add_node(1);
       for (NodeId id = 2; id < 300; ++id) graph.add_node(id);
       editor.reads({input}).after({299});  // Must survive vector reallocation.
       require(graph.nodes.front().inputs == std::vector<SlotId>({input}) &&
                   graph.nodes.front().after == std::vector<NodeId>({299}),
               "editor no longer refers to its id");
       graph.remove_node(1);
       rejects<std::out_of_range>([&] { editor.reads({input}); });
       graph.add_node(1);
       editor.reads({input});  // Editors select an id, not a historical node instance.
       require(graph.nodes.back().inputs == std::vector<SlotId>({input}),
               "editor did not select recreated id");
     }},
    {"remove", [] {
       Graph graph;
       const auto input = graph.add_input();
       const auto output = graph.add_slot();
       graph.add_node(2).reads({input}).writes({output});
       graph.add_node(3).after({2});
       graph.set_outputs({output});
       rejects<std::invalid_argument>([&] { graph.remove_slot(input); });
       rejects<std::invalid_argument>([&] { graph.remove_slot(output); });
       rejects<std::invalid_argument>([&] { graph.remove_node(2); });
       graph.set_inputs({});
       graph.set_outputs({});
       rejects<std::invalid_argument>([&] { graph.remove_slot(input); });
       rejects<std::invalid_argument>([&] { graph.remove_slot(output); });
       graph.node(3).after({});
       graph.remove_node(2);
       const auto incomplete = prepare(graph);
       require(!incomplete.ok() && incomplete.error->code == ErrorCode::missing_producer,
               "removing a producer must not silently remove its tensor");
       graph.remove_slot(input);
       graph.remove_slot(output);
       graph.remove_node(3);
       require(graph.nodes.empty() && graph.slots.empty() && prepare(graph).ok(),
               "explicit graph cleanup failed");
       rejects<std::out_of_range>([&] { graph.remove_node(99); });
       rejects<std::out_of_range>([&] { graph.remove_slot(99); });
     }},
    {"global_validation", [] {
       Graph graph;
       const auto output = graph.add_slot();
       graph.add_node(2).writes({output});
       graph.add_node(3).writes({output});
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
       const auto input = graph.add_input(tensor);
       const auto output = graph.add_slot(tensor);
       graph.add_node(10).reads({input}).writes({output}).bind(binding);
       graph.set_outputs({output});
       const auto result = prepare_program(graph, {entry});
       require(result.ok() && result.value->tensors.at(output).info.span_bytes == 24 &&
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
