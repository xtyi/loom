#include "cpu_graph.h"
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: loom-cpu-demo /path/to/libloom-example-add.so /path/to/libloom-example-relu.so\n";
    return 2;
  }
  try {
    const loom::examples::CpuGraph example;
    alignas(16) const float bias[] = {1, 1, 1, 1, 1, 1};
    alignas(16) const float input[] = {-3, -2, -1, 0, 1, 2};
    loom::cpu::Session session(example.graph, example.entries,
        {{{"add", "cpu"}, argv[1]}, {{"relu_scale", "cpu"}, argv[2]}},
        {{1, {bias, sizeof(bias)}}});
    const auto result = session.run({{input, sizeof(input)}});
    const auto* output = static_cast<const float*>(result.outputs.at(0).data.get());
    std::cout << "relu((input + bias)) * 2:";
    const float expected[] = {0, 0, 0, 2, 4, 6};
    for (std::size_t i = 0; i < 6; ++i) {
      if (output[i] != expected[i]) throw std::runtime_error("unexpected CPU result");
      std::cout << ' ' << output[i];
    }
    std::cout << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
