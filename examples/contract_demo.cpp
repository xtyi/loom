#include "plan/program.h"

#include <iostream>

int main() {
  using namespace loom::plan;
  TensorDesc full;
  full.dtype = DType::f32;
  full.shape = {2, 3};
  full.alignment = 16;
  TensorDesc half = full;
  half.dtype = DType::f16;

  EntryContract cast;
  cast.ref = {"cast", "cpu"};
  cast.symbol = "example_cast";
  cast.inputs = {{"x", full}};
  cast.outputs = {{"y", half}};
  cast.numerical_contract = "f32-to-f16-rne/v1";

  EntryContract compute;
  compute.ref = {"compute", "cpu"};
  compute.symbol = "example_compute";
  compute.inputs = {{"x", half}};
  compute.outputs = {{"y", half}};
  compute.numerical_contract = "f16-io-f32-accum/v1";
  compute.workspace_bytes = 64;
  compute.workspace_alignment = 16;
  const std::vector<EntryContract> entries{cast, compute};

  // 节点声明所需契约；这里与入口来自同一示例，实际可由离线构建工具提供。
  EntryBinding cast_binding;
  cast_binding.ref = cast.ref;
  cast_binding.numerical_contract = "f32-to-f16-rne/v1";
  EntryBinding compute_binding;
  compute_binding.ref = compute.ref;
  compute_binding.numerical_contract = "f16-io-f32-accum/v1";

  Graph graph;
  const auto input = graph.add_input(0, full);
  const auto converted = graph.add_slot(1, half);
  const auto output = graph.add_slot(2, half);
  graph.add_node(10).reads({converted}).writes({output}).bind(compute_binding);
  graph.add_node(30).reads({input}).writes({converted}).bind(cast_binding);
  graph.set_outputs({output});
  const auto result = prepare_program(graph, entries);
  if (!result.ok()) {
    const auto& error = *result.error;
    std::cerr << error.message << " (" << error.field << ")\n";
    return 1;
  }
  std::cout << "Validated metadata only; no kernel loaded or executed.\n";
  for (const auto& [id, tensor] : result.value->tensors) {
    std::cout << "slot " << id << ": " << dtype_name(tensor.info.desc.dtype)
              << ", span=" << tensor.info.span_bytes << " bytes\n";
  }
  for (const auto& call : result.value->calls) {
    std::cout << "node " << call.node << ": " << call.entry.ref.kernel << '/'
              << call.entry.ref.implementation << ", symbol=" << call.entry.symbol
              << ", contract=" << call.entry.numerical_contract
              << ", workspace=" << call.entry.workspace_bytes << " bytes\n";
  }
}
