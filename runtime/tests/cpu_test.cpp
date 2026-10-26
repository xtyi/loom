#include "examples/cpu_graph.h"

#include <cstring>
#include <dlfcn.h>
#include <functional>
#include <iostream>
#include <limits>
#include <string>

using loom::cpu::BufferView;
using loom::cpu::Error;
using loom::cpu::ErrorCode;
using loom::cpu::KernelLibrary;
using loom::cpu::Session;
using loom::examples::CpuGraph;
using loom::plan::DType;
using loom::plan::TensorDesc;

namespace {
std::string add_path, relu_path, fixture_path;
void require(bool value, const std::string& message) {
  if (!value) throw std::runtime_error(message);
}
std::vector<KernelLibrary> libraries() {
  return {{{"add", "cpu"}, add_path}, {{"relu_scale", "cpu"}, relu_path}};
}
struct Fixture {
  CpuGraph example{};
  alignas(16) float input[6] = {-3, -2, -1, 0, 1, 2};
  alignas(16) float bias[6] = {1, 1, 1, 1, 1, 1};
  std::map<loom::plan::SlotId, BufferView> constants() const {
    return {{1, {bias, sizeof(bias)}}};
  }
  BufferView view() const { return {input, sizeof(input)}; }
};
template <class Fn>
Error rejects(ErrorCode code, Fn fn) {
  try { fn(); }
  catch (const Error& error) {
    require(error.code == code, "wrong error category: " + std::string(error.what()));
    require(*error.what() != '\0', "missing diagnostic");
    return error;
  }
  throw std::runtime_error("invalid operation accepted");
}
void values(const loom::cpu::OutputTensor& output, const std::vector<float>& expected) {
  require(output.info.elements == expected.size(), "wrong element count");
  const auto* data = static_cast<const float*>(output.data.get());
  const auto stride = output.info.desc.shape.empty() ? 1 : output.info.desc.strides[0];
  for (std::size_t i = 0; i < expected.size(); ++i) {
    require(data[i * stride] == expected[i], "wrong numerical output at " + std::to_string(i));
  }
}
const std::map<std::string, std::function<void()>> cases{
  {"execute", [] {
    Fixture f;
    Session session(f.example.graph, f.example.entries, libraries(), f.constants());
    const auto result = session.run({f.view()});
    values(result.outputs.at(0), {0, 0, 0, 2, 4, 6});
    require(result.outputs[0].slot == 3, "wrong output slot");
    require(reinterpret_cast<uintptr_t>(result.outputs[0].data.get()) % 16 == 0,
            "output alignment violated");
    require(f.input[0] == -3 && f.bias[0] == 1, "kernel modified external data");
  }},
  {"lifetime", [] {
    const auto result = [] {
      Fixture f;
      Session session(f.example.graph, f.example.entries, libraries(), f.constants());
      f.bias[0] = 100; // Constant snapshot must be independent of caller storage.
      f.example.graph.nodes.clear();
      f.example.entries.clear();
      const auto first = session.run({f.view()});
      f.input[0] = 10;
      const auto second = session.run({f.view()});
      values(second.outputs[0], {22, 0, 0, 2, 4, 6});
      values(first.outputs[0], {0, 0, 0, 2, 4, 6});
      return first;
    }(); // Inputs, weights, metadata, dynamic libraries and Session destroyed.
    values(result.outputs[0], {0, 0, 0, 2, 4, 6});
  }},
  {"output_order", [] {
    Fixture f;
    f.example.graph.set_outputs({3, 2, 0, 1, 3});
    Session session(f.example.graph, f.example.entries, libraries(), f.constants());
    const auto result = session.run({f.view()});
    require(result.outputs.size() == 5, "output count changed");
    values(result.outputs[0], {0, 0, 0, 2, 4, 6});
    values(result.outputs[1], {-2, -1, 0, 1, 2, 3});
    values(result.outputs[2], {-3, -2, -1, 0, 1, 2});
    values(result.outputs[3], {1, 1, 1, 1, 1, 1});
    require(result.outputs[0].data == result.outputs[4].data, "repeated output not preserved");
    f.input[0] = 100;
    values(result.outputs[2], {-3, -2, -1, 0, 1, 2});
  }},
  {"shapes", [] {
    for (const auto& desc : std::vector<TensorDesc>{
        {DType::f32, {}, {}, 0, 16}, {DType::f32, {3}, {2}, 0, 16},
        {DType::f32, {0}, {}, 0, 16}}) {
      const CpuGraph example(desc);
      alignas(16) const float input[] = {2, 99, -2, 99, 4};
      alignas(16) const float bias[] = {1, 99, 1, 99, 1};
      const bool empty = !desc.shape.empty() && desc.shape[0] == 0;
      Session session(example.graph, example.entries, libraries(),
                      {{1, {empty ? nullptr : bias, empty ? 0 : sizeof(bias)}}});
      const auto result = session.run({{empty ? nullptr : input, empty ? 0 : sizeof(input)}});
      if (empty) {
        values(result.outputs[0], {});
        require(!result.outputs[0].data, "empty tensor allocated storage");
      } else if (desc.shape.empty()) values(result.outputs[0], {6});
      else {
        values(result.outputs[0], {6, 0, 10});
        require(result.outputs[0].info.span_bytes == 20, "wrong strided span");
        const auto* out = static_cast<const float*>(result.outputs[0].data.get());
        require(out[1] == 0 && out[3] == 0, "output padding contains uninitialized data");
      }
    }
  }},
  {"bindings", [] {
    Fixture f;
    Session session(f.example.graph, f.example.entries, libraries(), f.constants());
    rejects(ErrorCode::binding, [&] { session.run({}); });
    rejects(ErrorCode::binding, [&] { session.run({f.view(), f.view()}); });
    auto e = rejects(ErrorCode::capacity, [&] { session.run({{f.input, 23}}); });
    require(e.slot == 0, "missing capacity slot");
    rejects(ErrorCode::capacity, [&] { session.run({{nullptr, 24}}); });
    rejects(ErrorCode::alignment, [&] {
      session.run({{reinterpret_cast<const char*>(f.input) + 1, 24}});
    });
    const auto overflow = std::numeric_limits<uintptr_t>::max() - 15;
    rejects(ErrorCode::capacity, [&] { session.run({{reinterpret_cast<void*>(overflow), 24}}); });
    // Failed preflight must not poison the Session.
    values(session.run({f.view()}).outputs[0], {0, 0, 0, 2, 4, 6});
  }},
  {"constants", [] {
    Fixture f;
    rejects(ErrorCode::binding, [&] { Session s(f.example.graph, f.example.entries, libraries()); });
    rejects(ErrorCode::binding, [&] {
      Session s(f.example.graph, f.example.entries, libraries(), {{0, f.view()}});
    });
    rejects(ErrorCode::capacity, [&] {
      Session s(f.example.graph, f.example.entries, libraries(), {{1, {f.bias, 1}}});
    });
    rejects(ErrorCode::alignment, [&] {
      Session s(f.example.graph, f.example.entries, libraries(), {{1, {f.bias + 1, 24}}});
    });
    const auto extra = f.example.graph.add_external(f.example.graph.slots[1].tensor);
    rejects(ErrorCode::alias, [&] {
      Session s(f.example.graph, f.example.entries, libraries(),
                {{1, {f.bias, 24}}, {extra, {f.bias, 24}}});
    });
  }},
  {"input_order_alias", [] {
    Fixture f;
    // Two graph inputs, no preloaded weights; non-sorted interface order.
    f.example.graph.set_inputs({1, 0});
    f.example.graph.set_outputs({0, 1, 3});
    Session session(f.example.graph, f.example.entries, libraries());
    rejects(ErrorCode::alias, [&] { session.run({f.view(), f.view()}); });
    alignas(16) float storage[12]{};
    rejects(ErrorCode::alias, [&] { session.run({{storage, 24}, {storage + 4, 24}}); });
    const auto result = session.run({{f.bias, sizeof(f.bias)}, f.view()});
    values(result.outputs[0], {-3, -2, -1, 0, 1, 2});
    values(result.outputs[1], {1, 1, 1, 1, 1, 1});
    values(result.outputs[2], {0, 0, 0, 2, 4, 6});
  }},
  {"program", [] {
    Fixture f;
    f.example.graph.slots[0].tensor.dtype = DType::invalid;
    const auto e = rejects(ErrorCode::invalid_program, [&] {
      Session s(f.example.graph, f.example.entries, {}, f.constants());
    });
    require(e.diagnostic.has_value() && e.slot == 0, "prepare diagnostic lost");
    f = Fixture{};
    const auto e2 = rejects(ErrorCode::invalid_program, [&] {
      Session s(f.example.graph, f.example.entries, libraries(), f.constants(),
                std::vector<loom::plan::NodeId>{10, 30});
    });
    require(e2.diagnostic->code == loom::plan::ErrorCode::invalid_schedule, "order unchecked");
  }},
  {"libraries", [] {
    Fixture f;
    rejects(ErrorCode::invalid_library, [&] {
      Session s(f.example.graph, f.example.entries, {}, f.constants());
    });
    auto libs = libraries();
    libs.push_back(libs[0]);
    rejects(ErrorCode::invalid_library, [&] {
      Session s(f.example.graph, f.example.entries, libs, f.constants());
    });
    libs = libraries(); libs.push_back({{"unused", "cpu"}, add_path});
    rejects(ErrorCode::invalid_library, [&] {
      Session s(f.example.graph, f.example.entries, libs, f.constants());
    });
    libs = libraries(); libs[0].path = std::string("a\0b", 3);
    rejects(ErrorCode::invalid_library, [&] {
      Session s(f.example.graph, f.example.entries, libs, f.constants());
    });
    libs = libraries(); libs[0].path += ".does-not-exist";
    const auto e = rejects(ErrorCode::load, [&] {
      Session s(f.example.graph, f.example.entries, libs, f.constants());
    });
    require(e.node == 30, "load error node missing");
    libs = libraries(); libs[0].path = relu_path;
    rejects(ErrorCode::contract, [&] {
      Session s(f.example.graph, f.example.entries, libs, f.constants());
    });
  }},
  {"empty", [] {
    Session empty({}, {}, {});
    require(empty.run({}).outputs.empty(), "empty run failed");
    loom::plan::Graph graph;
    const auto input = graph.add_input({DType::i32, {}});
    graph.set_outputs({input});
    Session passthrough(graph, {}, {});
    const int32_t value = 123;
    const auto result = passthrough.run({{&value, sizeof(value)}});
    require(*static_cast<const int32_t*>(result.outputs[0].data.get()) == 123,
            "non-f32 pass-through failed");
  }},
  {"kernel_params", [] {
    Fixture f;
    // Use valid scalar descriptor: params remain opaque to prepare_program.
    f.example = CpuGraph({DType::f32, {}, {}, 0, 16}, std::numeric_limits<float>::quiet_NaN());
    Session session(f.example.graph, f.example.entries, libraries(), f.constants());
    const auto e = rejects(ErrorCode::kernel, [&] { session.run({f.view()}); });
    require(e.node == 10 && e.kernel_status == 3, "kernel status/context lost");
  }},
  {"fail_fast", [] {
    Fixture f;
    f.example.graph.node(10).reads({2, 1}).bind({{"add", "cpu"}, "f32-add/v1", {}});
    f.example.entries.resize(1);
    // Hold a separate reference to observe invocations across failed runs.
    void* handle = dlopen(fixture_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    require(handle != nullptr, "cannot open test probe");
    const std::shared_ptr<void> guard(handle, [](void* h) { dlclose(h); });
    const auto calls = reinterpret_cast<int (*)(void)>(dlsym(handle, "lm_test_calls"));
    require(calls != nullptr && calls() == 0, "invalid probe");
    Session session(f.example.graph, f.example.entries,
                    {{{"add", "cpu"}, fixture_path}}, f.constants());
    rejects(ErrorCode::capacity, [&] { session.run({{f.input, 1}}); });
    require(calls() == 0, "executed kernel before validating all buffers");
    for (int i = 1; i <= 2; ++i) {
      const auto e = rejects(ErrorCode::kernel, [&] { session.run({f.view()}); });
      require(e.node == 30 && e.kernel_status == 37, "wrong failed node/status");
      require(calls() == i, "executed node after kernel failure");
    }
  }},
  {"kernel_protocol", [] {
    Fixture f;
    auto libs = libraries(); libs[0].path = fixture_path;
    Session session(f.example.graph, f.example.entries, libs, f.constants());
    const auto e = rejects(ErrorCode::kernel_protocol, [&] { session.run({f.view()}); });
    require(e.node == 30 && e.kernel_status == -7, "invalid kernel status lost");
  }}
};

const std::map<std::string, ErrorCode> caps_cases{
  {"caps_version", ErrorCode::abi}, {"caps_truncated", ErrorCode::abi},
  {"caps_backend", ErrorCode::unsupported}, {"caps_async", ErrorCode::unsupported},
  {"caps_identity", ErrorCode::contract}, {"caps_numerics", ErrorCode::contract},
  {"caps_alignment", ErrorCode::abi}, {"caps_ports", ErrorCode::contract},
  {"caps_workspace", ErrorCode::contract}, {"caps_params", ErrorCode::contract},
  {"missing_entry", ErrorCode::symbol}, {"missing_caps", ErrorCode::symbol},
  {"caps_null", ErrorCode::abi}, {"caps_flags", ErrorCode::unsupported},
  {"caps_stronger_alignment", ErrorCode::contract}
};
} // namespace

int main(int argc, char** argv) {
  if (argc < 4) return 2;
  add_path = argv[2]; relu_path = argv[3];
  if (argc > 4) fixture_path = argv[4];
  try {
    if (caps_cases.count(argv[1])) {
      Fixture f;
      auto libs = libraries(); libs[0].path = fixture_path;
      const auto e = rejects(caps_cases.at(argv[1]), [&] {
        Session s(f.example.graph, f.example.entries, libs, f.constants());
      });
      require(e.node == 30, "missing caps error node");
    } else cases.at(argv[1])();
  } catch (const std::exception& error) {
    std::cerr << argv[1] << ": " << error.what() << '\n';
    return 1;
  }
}
