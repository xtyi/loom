#include "nv/backend.h"
#include "nv/package.h"
#include "nv/session.h"
#include "cpu/package.h"
#include "examples/elementwise_graph.h"
#include <cuda_runtime_api.h>
#include <cstring>
#include <dlfcn.h>
#include <functional>
#include <iostream>
#include <limits>

namespace {
using namespace loom;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
execution::Error rejects(execution::ErrorCode code, const std::function<void()>& fn) {
  try { fn(); } catch (const execution::Error& error) {
    require(error.code == code, error.what());
    return error;
  }
  throw std::runtime_error("invalid operation accepted");
}
void check(cudaError_t status) { require(status == cudaSuccess, cudaGetErrorString(status)); }
}
int main(int argc, char** argv) {
  if (argc != 7) return 2;
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
  try {
    const std::string test = argv[1];
    if (test == "resources") {
      int previous;
      check(cudaGetDevice(&previous));
      auto backend = nv::make_backend(count > 1 ? (previous + 1) % count : previous);
      int current;
      check(cudaGetDevice(&current));
      require(current == previous, "backend construction changed current device");
      auto device = backend->allocate(64, 4096);
      require(reinterpret_cast<std::uintptr_t>(device.get()) % 4096 == 0, "device alignment lost");
      require(!backend->allocate(0, 16), "empty allocation is not null");
      rejects(execution::ErrorCode::capacity, [&] {
        backend->allocate(std::numeric_limits<std::size_t>::max(), 4096);
      });
      alignas(4096) unsigned char input[64], output[64];
      std::memset(input, 123, sizeof(input));
      backend->upload(device.get(), input, sizeof(input));
      backend->download(output, device.get(), sizeof(output));
      require(std::memcmp(input, output, sizeof(input)) == 0, "device round trip failed");
      backend->zero(device.get(), sizeof(input));
      backend->download(output, device.get(), sizeof(output));
      for (auto byte : output) require(byte == 0, "device zero failed");
      backend.reset();
      device.reset(); // allocation owns its device even after backend destruction
      check(cudaGetDevice(&current));
      require(current == previous, "backend operations changed current device");
      rejects(execution::ErrorCode::device, [&] { nv::make_backend(count); });
      rejects(execution::ErrorCode::device, [&] { nv::make_backend(-1); });
    } else if (test == "package") {
      auto package = io::load_package(argv[6]);
      auto session = nv::make_session(package);
      const auto input = io::load_tensor(package.root / "inputs/0.bin", package.program.tensors.at(0).info);
      const auto result = session->run({{input.data.get(), input.bytes}});
      session.reset();
      const float expected[] = {0, 0, 0, 2, 4, 6};
      require(std::memcmp(result.outputs.at(0).data.get(), expected, sizeof(expected)) == 0, "CUDA package result differs from CPU oracle");
      rejects(execution::ErrorCode::unsupported, [&] { cpu::make_session(package); });
      package.compute_capability += 1;
      rejects(execution::ErrorCode::unsupported, [&] { nv::make_session(package); });
      package.compute_capability -= 1;
      package.entries[0].numerical_contract = "changed";
      rejects(execution::ErrorCode::invalid_program, [&] { nv::make_session(package); });
      require(nv::load_package(argv[6])->run({{input.data.get(), input.bytes}}).outputs.size() == 1, "CUDA convenience loader failed");
    } else {
      examples::ElementwiseGraph example({plan::DType::f32, {6}, {}, 0, 16}, 2.f, "cuda");
      alignas(16) const float input[] = {-3, -2, -1, 0, 1, 2};
      alignas(16) const float bias[] = {1, 1, 1, 1, 1, 1};
      std::map<plan::SlotId, execution::BufferView> constants{{1, {bias, sizeof(bias)}}};
      if (test == "backend") {
        const auto error = rejects(execution::ErrorCode::unsupported, [&] {
          nv::Session session(example.graph, example.entries,
              {{{"add", "cuda"}, argv[4]}, {{"relu_scale", "cuda"}, argv[3]}}, constants);
        });
        require(error.node == 30, "backend error lost node");
      } else if (test == "failure" || test == "async_error") {
        void* raw = dlopen(argv[5], RTLD_NOW | RTLD_LOCAL);
        require(raw != nullptr, "fixture load failed");
        const std::shared_ptr<void> library(raw, [](void* h) { dlclose(h); });
        const auto calls = reinterpret_cast<int (*)()>(dlsym(raw, "lm_test_calls"));
        const auto done = reinterpret_cast<int (*)()>(dlsym(raw, "lm_test_completed"));
        const auto status = reinterpret_cast<void (*)(int)>(dlsym(raw, "lm_test_status"));
        require(calls && done && status, "fixture symbols missing");
        example.graph.node(10).reads({2, 1}).bind({{"add", "cuda"}, "f32-add/v1", {}});
        example.entries.resize(1);
        nv::Session session(example.graph, example.entries, {{{"add", "cuda"}, argv[5]}}, constants);
        if (test == "async_error") {
          status(0);
          const auto error = rejects(execution::ErrorCode::device, [&] { session.run({{input, sizeof(input)}}); });
          require(error.node == 30 && calls() == 1, "async error lost node or continued execution");
        } else {
          for (const int code : {37, -7}) {
            status(code);
            const auto error = rejects(code == 37 ? execution::ErrorCode::kernel : execution::ErrorCode::kernel_protocol,
                [&] { session.run({{input, sizeof(input)}}); });
            require(error.node == 30 && error.kernel_status == code, "launcher failure context lost");
            require(done(), "error returned before stream completed");
          }
          require(calls() == 2, "executed node after failure");
        }
      } else throw std::runtime_error("unknown test");
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
