#ifndef LOOM_EXAMPLES_ELEMENTWISE_GRAPH_H
#define LOOM_EXAMPLES_ELEMENTWISE_GRAPH_H

#include "plan/program.h"
#include <cstring>

namespace loom::examples {

// Example-only assembly; the runtime does not know add/relu semantics.
struct ElementwiseGraph {
  plan::Graph graph;
  std::vector<plan::EntryContract> entries;

  explicit ElementwiseGraph(plan::TensorDesc tensor = {plan::DType::f32, {6}, {}, 0, 16},
                    float scale = 2.f, const std::string& implementation = "cpu") {
    plan::EntryContract add;
    add.ref = {"add", implementation};
    add.symbol = "lm_ku_entry_exp_v1";
    add.inputs = {{"x", tensor}, {"bias", tensor}};
    add.outputs = {{"y", tensor}};
    add.numerical_contract = "f32-add/v1";

    plan::EntryContract relu;
    relu.ref = {"relu_scale", implementation};
    relu.symbol = "lm_ku_entry_exp_v1";
    relu.inputs = {{"x", tensor}};
    relu.outputs = {{"y", tensor}};
    relu.numerical_contract = "f32-relu-scale/v1";
    relu.params_bytes = sizeof(float);
    relu.workspace_bytes = 16;
    relu.workspace_alignment = 16;
    entries = {add, relu};

    const auto x = graph.add_input(tensor);
    const auto bias = graph.add_external(tensor);
    const auto sum = graph.add_slot(tensor);
    const auto output = graph.add_slot(tensor);
    plan::EntryBinding scale_binding{relu.ref, relu.numerical_contract, {}};
    scale_binding.params.resize(sizeof(scale));
    std::memcpy(scale_binding.params.data(), &scale, sizeof(scale));
    // Consumer first deliberately: prepare_program determines execution order.
    graph.add_node(10).reads({sum}).writes({output}).bind(scale_binding);
    graph.add_node(30).reads({x, bias}).writes({sum}).bind({add.ref, add.numerical_contract, {}});
    graph.set_outputs({output});
  }
};
} // namespace loom::examples
#endif
