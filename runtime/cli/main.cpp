#include "io/package.h"
#ifdef LOOM_CLI_CPU
#include "cpu/package.h"
#endif

#include <fstream>
#include <iostream>

namespace {
void usage() {
  std::cerr << "Usage:\n"
               "  loom-cli inspect PACKAGE_DIR\n"
               "  loom-cli validate PACKAGE_DIR\n"
               "  loom-cli run PACKAGE_DIR OUTPUT_DIR [INPUT.bin ...]\n";
}
} // namespace

int main(int argc, char** argv) {
  if (argc < 3) { usage(); return 2; }
  const std::string command = argv[1];
  if ((command != "inspect" && command != "validate" && command != "run") ||
      (command != "run" && argc != 3) || (command == "run" && argc < 4)) {
    usage(); return 2;
  }
  try {
#ifndef LOOM_CLI_CPU
    if (command == "run") throw std::runtime_error("CPU execution requires -DLOOM_ENABLE_EXPERIMENTAL_ABI=ON");
#endif
    const auto package = loom::io::load_package(argv[2]);
    if (command == "validate") {
      std::cout << "Valid package metadata and constant files; kernels have not been loaded or executed.\n";
    } else if (command == "inspect") {
      std::cout << "Package: " << package.name << " (" << loom::io::package_format << ")\n";
      for (const auto& [id, tensor] : package.program.tensors) {
        std::cout << "slot " << id << ": " << loom::plan::dtype_name(tensor.info.desc.dtype)
                  << ", span=" << tensor.info.span_bytes << " bytes\n";
      }
      std::cout << "Inputs:";
      for (auto id : package.program.inputs) std::cout << ' ' << id;
      std::cout << "\nOutputs:";
      for (auto id : package.program.outputs) std::cout << ' ' << id;
      std::cout << "\nSchedule:";
      for (const auto& call : package.program.calls) std::cout << ' ' << call.node;
      std::cout << '\n';
      for (const auto& call : package.program.calls) {
        std::cout << "node " << call.node << ": " << call.entry.ref.kernel << '/'
                  << call.entry.ref.implementation << ", contract=" << call.entry.numerical_contract
                  << ", workspace=" << call.entry.workspace_bytes << " bytes\n";
      }
    }
#ifdef LOOM_CLI_CPU
    else {
      if (static_cast<std::size_t>(argc - 4) != package.program.inputs.size()) {
        throw std::runtime_error("input files must follow graph.inputs order and cover every input");
      }
      std::vector<loom::io::TensorData> storage;
      std::vector<loom::cpu::BufferView> inputs;
      for (std::size_t i = 0; i < package.program.inputs.size(); ++i) {
        const auto slot = package.program.inputs[i];
        storage.push_back(loom::io::load_tensor(argv[4 + i], package.program.tensors.at(slot).info));
        inputs.push_back({storage.back().data.get(), storage.back().bytes});
      }
      std::filesystem::create_directories(argv[3]);
      for (std::size_t i = 0; i < package.program.outputs.size(); ++i) {
        const auto path = std::filesystem::path(argv[3]) / (std::to_string(i) + ".bin");
        if (std::filesystem::exists(path)) throw std::runtime_error("output already exists: " + path.string());
      }
      const auto result = loom::cpu::make_session(package)->run(inputs);
      for (std::size_t i = 0; i < result.outputs.size(); ++i) {
        const auto& tensor = result.outputs[i];
        const auto path = std::filesystem::path(argv[3]) / (std::to_string(i) + ".bin");
        // Avoid silently clobbering an earlier inference's results.
        if (std::filesystem::exists(path)) throw std::runtime_error("output already exists: " + path.string());
        std::ofstream out(path, std::ios::binary);
        if (!out || (tensor.info.span_bytes &&
            !out.write(static_cast<const char*>(tensor.data.get()),
                       static_cast<std::streamsize>(tensor.info.span_bytes)))) {
          throw std::runtime_error("cannot write output: " + path.string());
        }
        out.close();
        if (!out) throw std::runtime_error("cannot finish output: " + path.string());
        std::cout << "output " << i << ": slot " << tensor.slot << ", "
                  << tensor.info.span_bytes << " bytes -> " << path.string() << '\n';
      }
    }
#endif
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
