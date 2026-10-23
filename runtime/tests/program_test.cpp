#include "plan/program.h"

#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>

using namespace loom::plan;

namespace {

void require(bool value, const std::string& message) {
  if (!value) throw std::runtime_error(message);
}

struct Fixture {
  TensorDesc full{DType::f32, {2, 3}, {}, 0, 16};
  TensorDesc half{DType::f16, {2, 3}, {}, 0, 16};
  TensorDesc bias{DType::f32, {3}, {}, 0, 16};
  Graph graph{
      {{0, true, full}, {1, false, half}, {2, true, bias}, {3, false, half}},
      {{10, {1, 2}, {3}, {}, {{"compute", "cpu"}, "f16-f32acc/v1", {7, 9}}},
       {30, {0}, {1}, {}, {{"cast", "cpu"}, "rne-f32-f16/v1", {}}}},
      {0}, {3}};
  std::vector<EntryContract> entries{
      {{"cast", "cpu"}, "test_cast", {{"x", full}}, {{"y", half}},
       "rne-f32-f16/v1"},
      {{"compute", "cpu"}, "test_compute", {{"x", half}, {"bias", bias}},
       {{"y", half}}, "f16-f32acc/v1", 2, 64, 32}};
  ProgramResult run() const { return prepare_program(graph, entries); }
};

void bad(const ProgramResult& result, ErrorCode code, const std::string& field = {},
         std::optional<NodeId> node = {}, std::optional<SlotId> slot = {}) {
  require(!result.ok() && result.error.has_value(), "invalid program accepted");
  require(result.error->code == code, "wrong error category");
  require(!result.error->message.empty(), "missing diagnostic");
  if (!field.empty()) require(result.error->field == field, "wrong error field");
  if (node) require(result.error->node == node, "wrong error node");
  if (slot) require(result.error->slot == slot, "wrong error slot");
}

const std::map<std::string, std::function<void()>> cases{
    {"mixed_plan", [] {
       const auto result = Fixture{}.run();
       require(result.ok(), result.error ? result.error->message : "prepare failed");
       require(!result.error, "success also reported failure");
       const auto& plan = *result.value;
       require(plan.calls.size() == 2 && plan.calls[0].node == 30 &&
                   plan.calls[1].node == 10, "lost dependency order");
       require(plan.tensors.at(0).info.span_bytes == 24 &&
                   plan.tensors.at(1).info.span_bytes == 12, "wrong tensor size");
       require(plan.tensors.at(0).external && !plan.tensors.at(1).external,
               "lost input ownership classification");
       const auto& call = plan.calls[1];
       require(call.entry.symbol == "test_compute" && call.entry.workspace_bytes == 64 &&
                   call.entry.workspace_alignment == 32, "lost entry requirements");
       require(call.params == std::vector<std::uint8_t>({7, 9}) &&
                   call.inputs == std::vector<SlotId>({1, 2}) &&
                   call.outputs == std::vector<SlotId>({3}), "lost positional binding");
       require(plan.outputs == std::vector<SlotId>({3}), "lost graph outputs");
       require(plan.inputs == std::vector<SlotId>({0}) && plan.tensors.at(2).external,
               "preloaded bias must not become a per-call input");
     }},
    {"graph_inputs", [] {
       Fixture f;
       f.graph.inputs = {2, 0};
       const auto result = f.run();
       require(result.ok() && result.value->inputs == std::vector<SlotId>({2, 0}),
               "graph input order must be preserved, not sorted or inferred");
       f.graph.inputs = {1};
       bad(f.run(), ErrorCode::invalid_graph_input, {}, {}, 1);
       f.graph.inputs = {99};
       bad(f.run(), ErrorCode::unknown_slot, {}, {}, 99);
       f.graph.inputs = {0, 0};
       bad(f.run(), ErrorCode::duplicate_graph_input, {}, {}, 0);
     }},
    {"missing_metadata", [] {
       Fixture f;
       f.graph.slots[0].tensor = {};
       require(prepare(f.graph).ok(), "topology-only preparation must stay separate");
       bad(f.run(), ErrorCode::invalid_tensor, "dtype", {}, 0);
       f = Fixture{};
       f.graph.nodes[1].binding = {};
       bad(f.run(), ErrorCode::unknown_entry, "binding", 30);
     }},
    {"registry", [] {
       Fixture f;
       f.entries.push_back(f.entries[0]);
       bad(f.run(), ErrorCode::duplicate_entry, "entry");
       f = Fixture{};
       f.graph.nodes[1].binding.ref.implementation = "missing";
       bad(f.run(), ErrorCode::unknown_entry, "binding", 30);
       f = Fixture{};
       f.entries[1].symbol.clear();
       bad(f.run(), ErrorCode::invalid_entry);
       f = Fixture{};
       f.entries[0].ref.kernel = std::string("a\0b", 3);
       bad(f.run(), ErrorCode::invalid_entry);
     }},
    {"invalid_ports", [] {
       Fixture f;
       f.entries[1].inputs[1].name = "x";
       bad(f.run(), ErrorCode::invalid_entry);
       f = Fixture{};
       f.entries[0].outputs[0].name.clear();
       bad(f.run(), ErrorCode::invalid_entry);
       f = Fixture{};
       f.entries[0].outputs[0].tensor.shape = {-1};
       bad(f.run(), ErrorCode::invalid_entry);
       f = Fixture{};
       auto unused = f.entries[0];
       unused.ref.kernel = "unused";
       unused.inputs[0].tensor.dtype = DType::invalid;
       f.entries.push_back(unused);
       bad(f.run(), ErrorCode::invalid_entry);
     }},
    {"arity", [] {
       Fixture f;
       f.graph.nodes[0].inputs.pop_back();
       bad(f.run(), ErrorCode::contract_mismatch, "inputs.count", 10);
       f = Fixture{};
       f.entries[1].outputs.push_back({"extra", f.half});
       bad(f.run(), ErrorCode::contract_mismatch, "outputs.count", 10);
     }},
    {"port_order", [] {
       Fixture f;
       std::swap(f.graph.nodes[0].inputs[0], f.graph.nodes[0].inputs[1]);
       const auto result = f.run();
       bad(result, ErrorCode::contract_mismatch, "inputs[0].dtype", 10, 2);
       require(result.error->expected == "f16" && result.error->actual == "f32",
               "missing expected/actual values");
     }},
    {"dtype", [] {
       Fixture f;
       f.graph.slots[3].tensor.dtype = DType::f32;
       bad(f.run(), ErrorCode::contract_mismatch, "outputs[0].dtype", 10, 3);
       f = Fixture{};
       // A valid cast producer cannot bypass a consumer's incompatible contract.
       f.entries[1].inputs[0].tensor.dtype = DType::f32;
       bad(f.run(), ErrorCode::contract_mismatch, "inputs[0].dtype", 10, 1);
     }},
    {"shape", [] {
       Fixture f;
       f.graph.slots[0].tensor.shape = {6};  // Same bytes, different rank/shape.
       bad(f.run(), ErrorCode::contract_mismatch, "inputs[0].shape", 30, 0);
       f.graph.slots[0].tensor.shape = {3, 2};
       bad(f.run(), ErrorCode::contract_mismatch, "inputs[0].shape", 30, 0);
     }},
    {"layout", [] {
       Fixture f;
       f.graph.slots[0].tensor.layout = 7;
       bad(f.run(), ErrorCode::contract_mismatch, "inputs[0].layout", 30, 0);
       f.entries[0].inputs[0].tensor.layout = 7;
       require(f.run().ok(), "matching opaque labels rejected");
     }},
    {"strides", [] {
       Fixture f;
       f.graph.slots[0].tensor.strides = {3, 1};
       require(f.run().ok(), "explicit compact and omitted strides must match");
       f.graph.slots[0].tensor.strides = {4, 1};
       bad(f.run(), ErrorCode::contract_mismatch, "inputs[0].strides", 30, 0);
       f.entries[0].inputs[0].tensor.strides = {4, 1};
       require(f.run().ok(), "matching padded strides rejected");
     }},
    {"alignment", [] {
       Fixture f;
       f.graph.slots[0].tensor.alignment = 8;
       bad(f.run(), ErrorCode::contract_mismatch, "inputs[0].alignment", 30, 0);
       f.graph.slots[0].tensor.alignment = 32;
       require(f.run().ok(), "stronger slot alignment must satisfy entry");
       f.entries[1].workspace_alignment = 3;
       bad(f.run(), ErrorCode::invalid_entry);
     }},
    {"numerics", [] {
       Fixture f;
       f.graph.nodes[0].binding.numerical_contract = "f16-f16acc/v1";
       bad(f.run(), ErrorCode::contract_mismatch, "numerical_contract", 10);
       f.graph.nodes[0].binding.numerical_contract.clear();
       bad(f.run(), ErrorCode::contract_mismatch, "numerical_contract", 10);
       f = Fixture{};
       f.entries[1].numerical_contract.clear();
       bad(f.run(), ErrorCode::invalid_entry);
     }},
    {"params", [] {
       Fixture f;
       f.graph.nodes[0].binding.params.pop_back();
       bad(f.run(), ErrorCode::contract_mismatch, "params_bytes", 10);
       f.graph.nodes[0].binding.params = {1, 2, 3};
       bad(f.run(), ErrorCode::contract_mismatch, "params_bytes", 10);
     }},
    {"snapshot", [] {
       Fixture f;
       const auto result = f.run();
       require(result.ok(), "prepare failed");
       f.graph.inputs.clear();
       f.entries[1].symbol = "changed";
       f.graph.nodes[0].binding.params[0] = 0;
       f.graph.slots[0].tensor.shape[0] = 100;
       require(result.value->inputs == std::vector<SlotId>({0}) &&
                   result.value->calls[1].entry.symbol == "test_compute" &&
                   result.value->calls[1].params[0] == 7 &&
                   result.value->tensors.at(0).info.desc.shape[0] == 2,
               "prepared metadata aliases mutable input");
     }},
    {"topology", [] {
       Fixture f;
       bad(prepare_program(f.graph, f.entries, std::vector<NodeId>{10, 30}),
           ErrorCode::invalid_schedule);
       f.graph.nodes[1].after = {10};
       bad(f.run(), ErrorCode::cycle);
     }},
    {"empty", [] {
       require(prepare_program(Graph{}, {}).ok(), "empty program rejected");
       const Graph graph{{{0, true, {DType::f32, {}}}}, {}, {0}, {0}};
       const auto result = prepare_program(graph, {});
       require(result.ok() && result.value->calls.empty() &&
                   result.value->inputs == std::vector<SlotId>({0}) &&
                   result.value->outputs == std::vector<SlotId>({0}) &&
                   result.value->tensors.at(0).info.span_bytes == 4,
               "external pass-through scalar rejected");
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
