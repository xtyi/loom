#include "io/package.h"
#ifdef LOOM_TEST_CPU
#include "cpu/package.h"
#endif
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>

using nlohmann::json;
using namespace loom::io;
namespace fs = std::filesystem;
namespace {
std::string example_package;
void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
void write(const fs::path& path, const std::string& data) {
  fs::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary); out << data;
  if (!out) throw std::runtime_error("test file write failed");
}
json read(const fs::path& path) { std::ifstream in(path); return json::parse(in); }
void change(const fs::path& path, const std::function<void(json&)>& edit) {
  auto j = read(path); edit(j); write(path, j.dump());
}
struct Fixture {
  fs::path root;
  Fixture() {
    root = fs::temp_directory_path() / ("loom-package-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directory(root);
    const auto host = host_target();
    json tensor = {{"dtype","f32"},{"shape",{1}},{"alignment",64}};
    write(root/"package.json", json{{"format",package_format},{"name","fixture"},
      {"target",{{"backend","cpu"},{"abi","loom.cpu.exp.v1"},{"os",host.os},{"arch",host.arch},{"endianness",host.endianness},{"pointer_bits",host.pointer_bits}}},
      {"plan","plans/main.json"},{"kernels",{"kernels/id.json"}},
      {"constants",json::array({{{"slot",12},{"file","weights.bin"}}})}}.dump());
    write(root/"plans/main.json", json{{"format",plan_format},
      {"slots",json::array({{{"id",12},{"external",true},{"tensor",tensor}},{{"id",4},{"external",false},{"tensor",tensor}}})},
      {"nodes",json::array({{{"id",10},{"inputs",{12}},{"outputs",{4}},{"binding",{{"kernel","identity"},{"implementation","cpu"},{"numerical_contract","identity/v1"}}}}})},
      {"inputs",json::array()},{"outputs",{4}}}.dump());
    write(root/"kernels/id.json", json{{"format",kernel_format},{"kernel","identity"},{"implementation","cpu"},
      {"symbol","lm_ku_entry_exp_v1"},{"inputs",json::array({{{"name","x"},{"tensor",tensor}}})},
      {"outputs",json::array({{{"name","y"},{"tensor",tensor}}})},{"numerical_contract","identity/v1"},
      {"library","kernels/id.so"}}.dump());
    // Deliberately not a shared library: metadata loading must never dlopen.
    write(root/"kernels/id.so", "placeholder");
    const float constant = 7.f;
    write(root/"weights.bin", std::string(reinterpret_cast<const char*>(&constant), sizeof(constant)));
  }
  ~Fixture() { std::error_code ec; fs::remove_all(root, ec); }
};
Error rejects(ErrorCode code, const std::function<void()>& fn) {
  try { fn(); } catch (const Error& e) {
    require(e.code == code, std::string("wrong error: ") + e.what());
    require(!e.source.empty() && !e.field.empty(), "diagnostic context missing");
    return e;
  }
  throw std::runtime_error("invalid package accepted");
}

const std::map<std::string, std::function<void()>> cases{
  {"load", [] {
    Fixture f;
    const auto p = load_package(f.root);
    require(p.name == "fixture" && p.plan.graph.slots[0].id == 12, "metadata lost");
    require(p.program.calls[0].node == 10 && p.libraries[0].path.is_absolute(), "unprepared package");
    const auto& c = p.constants.at(12);
    require(c.bytes == 4 && reinterpret_cast<std::uintptr_t>(c.data.get()) % 64 == 0, "constant alignment lost");
    require(*static_cast<const float*>(c.data.get()) == 7.f, "wrong constant contents");
    write(f.root/"weights.bin", "xxxx");
    require(*static_cast<const float*>(c.data.get()) == 7.f, "constant snapshot aliases file");
  }},
  {"versions", [] {
    Fixture f;
    for (const auto* filename : {"package.json","plans/main.json","kernels/id.json"}) {
      const auto saved = read(f.root/filename);
      change(f.root/filename, [](json& j) { j["format"] = "future/99"; });
      rejects(ErrorCode::version, [&] { load_package(f.root); });
      write(f.root/filename, saved.dump());
      change(f.root/filename, [](json& j) { j["required_capabilities"] = {"async"}; });
      rejects(ErrorCode::capability, [&] { load_package(f.root); });
      write(f.root/filename, saved.dump());
    }
  }},
  {"target", [] {
    Fixture f;
    const auto saved = read(f.root/"package.json");
    for (const auto* field : {"abi","backend","os","arch","endianness"}) {
      change(f.root/"package.json", [&](json& j) { j["target"][field] = "different"; });
      rejects(ErrorCode::target, [&] { load_package(f.root); });
      write(f.root/"package.json", saved.dump());
    }
    change(f.root/"package.json", [](json& j) { j["target"]["pointer_bits"] = 1; });
    rejects(ErrorCode::target, [&] { load_package(f.root); });
  }},
  {"paths", [] {
    Fixture f;
    const auto saved = read(f.root/"package.json");
    for (const auto* name : {"../outside.json","plans/../../outside.json","/tmp/outside.json","C:\\outside.json"}) {
      change(f.root/"package.json", [&](json& j) { j["plan"] = name; });
      rejects(ErrorCode::path, [&] { load_package(f.root); });
      write(f.root/"package.json", saved.dump());
    }
    change(f.root/"kernels/id.json", [](json& j) { j["library"] = "../id.so"; });
    rejects(ErrorCode::path, [&] { load_package(f.root); });
  }},
  {"symlinks", [] {
    Fixture f, outside;
    std::error_code ec;
    fs::create_symlink(outside.root/"plans/main.json", f.root/"escaped.json", ec);
    if (ec) throw std::runtime_error("cannot create symlink test: " + ec.message());
    change(f.root/"package.json", [](json& j) { j["plan"] = "escaped.json"; });
    rejects(ErrorCode::path, [&] { load_package(f.root); });
  }},
  {"files", [] {
    Fixture f;
    const auto saved = read(f.root/"package.json");
    change(f.root/"package.json", [](json& j) { j["plan"] = "missing.json"; });
    rejects(ErrorCode::io, [&] { load_package(f.root); });
    write(f.root/"package.json", saved.dump());
    change(f.root/"package.json", [](json& j) { j["plan"] = "plans"; });
    rejects(ErrorCode::path, [&] { load_package(f.root); });
    rejects(ErrorCode::io, [&] { load_plan(f.root/"absent.json"); });
  }},
  {"contracts", [] {
    Fixture f;
    const auto saved = read(f.root/"kernels/id.json");
    for (const auto& edit : std::vector<std::function<void(json&)>>{
      [](json& j) { j["numerical_contract"] = "changed/v2"; },
      [](json& j) { j["inputs"][0]["tensor"]["dtype"] = "f16"; },
      [](json& j) { j["params_bytes"] = 4; },
      [](json& j) { j["workspace_alignment"] = 3; },
      [](json& j) { j["inputs"].push_back(j["inputs"][0]); }
    }) {
      change(f.root/"kernels/id.json", edit);
      const auto e = rejects(ErrorCode::program, [&] { load_package(f.root); });
      require(e.diagnostic.has_value(), "lost contract diagnostic");
      write(f.root/"kernels/id.json", saved.dump());
    }
    change(f.root/"kernels/id.json", [](json& j) { j["future_execution_flag"] = true; });
    rejects(ErrorCode::schema, [&] { load_package(f.root); });
  }},
  {"registry", [] {
    Fixture f;
    const auto saved = read(f.root/"package.json");
    change(f.root/"package.json", [](json& j) { j["kernels"] = json::array(); });
    rejects(ErrorCode::program, [&] { load_package(f.root); });
    write(f.root/"package.json", saved.dump());
    change(f.root/"package.json", [](json& j) { j["kernels"].push_back("kernels/id.json"); });
    rejects(ErrorCode::program, [&] { load_package(f.root); });
    write(f.root/"package.json", saved.dump());
    auto unused = read(f.root/"kernels/id.json"); unused["kernel"] = "unused";
    write(f.root/"kernels/unused.json", unused.dump());
    change(f.root/"package.json", [](json& j) { j["kernels"].push_back("kernels/unused.json"); });
    rejects(ErrorCode::binding, [&] { load_package(f.root); });
  }},
  {"constants", [] {
    Fixture f;
    const auto saved = read(f.root/"package.json");
    for (const auto& values : {json::array(), json::array({{{"slot",4},{"file","weights.bin"}}}),
      json::array({{{"slot",12},{"file","weights.bin"}},{{"slot",12},{"file","weights.bin"}}})}) {
      auto j = saved; j["constants"] = values; write(f.root/"package.json", j.dump());
      rejects(ErrorCode::binding, [&] { load_package(f.root); });
    }
    write(f.root/"package.json", saved.dump());
    for (const auto* bytes : {"", "abc", "abcde"}) {
      write(f.root/"weights.bin", bytes);
      rejects(ErrorCode::binding, [&] { load_package(f.root); });
    }
  }},
  {"empty", [] {
    Fixture f;
    auto j = read(f.root/"package.json"); j["kernels"] = json::array(); j["constants"] = json::array();
    write(f.root/"package.json", j.dump());
    write(f.root/"plans/main.json", json{{"format",plan_format},{"slots",json::array()},{"nodes",json::array()},
      {"inputs",json::array()},{"outputs",json::array()},{"schedule",{{"order",json::array()}}}}.dump());
    require(load_package(f.root).program.calls.empty(), "empty package rejected");
    json tensor = {{"dtype","f32"},{"shape",{0}},{"alignment",16}};
    change(f.root/"plans/main.json", [&](json& plan) {
      plan["slots"] = json::array({{{"id",99},{"external",true},{"tensor",tensor}}});
      plan["outputs"] = {99,99};
    });
    change(f.root/"package.json", [](json& p) { p["constants"] = json::array({{{"slot",99},{"file","empty.bin"}}}); });
    write(f.root/"empty.bin", "");
    auto p = load_package(f.root);
    require(p.constants.at(99).bytes == 0 && !p.constants.at(99).data, "zero-sized constant broken");
    require(p.program.outputs == std::vector<std::uint32_t>({99,99}), "repeated outputs lost");
  }},
#ifdef LOOM_TEST_CPU
  {"execute", [] {
    Fixture f;
    fs::copy(example_package, f.root, fs::copy_options::recursive | fs::copy_options::overwrite_existing);
    auto p = load_package(f.root);
    auto session = loom::cpu::make_session(p);
    const auto input = load_tensor(f.root/"inputs/0.bin", p.program.tensors.at(0).info);
    p = {}; // Session owns the necessary metadata, libraries, and constants.
    const auto result = session->run({{input.data.get(), input.bytes}});
    session.reset();
    const float expected[] = {0,0,0,2,4,6};
    require(std::memcmp(result.outputs[0].data.get(), expected, sizeof(expected)) == 0, "package execution failed");
    auto again = loom::cpu::load_package(f.root);
    require(again->run({{input.data.get(),input.bytes}}).outputs.size() == 1, "convenience loader failed");
  }},
  {"binary_caps", [] {
    Fixture f;
    fs::copy(example_package, f.root, fs::copy_options::recursive | fs::copy_options::overwrite_existing);
    change(f.root/"plans/main.json", [](json& j) { j["nodes"][0]["binding"]["numerical_contract"] = "other/v1"; });
    change(f.root/"kernels/relu_scale/ku.json", [](json& j) { j["numerical_contract"] = "other/v1"; });
    auto p = load_package(f.root); // JSON agrees; binary still declares original contract.
    try { loom::cpu::make_session(p); }
    catch (const loom::cpu::Error& e) { require(e.code == loom::cpu::ErrorCode::contract && e.node == 10, "wrong binary caps error"); return; }
    throw std::runtime_error("binary/metadata contract mismatch accepted");
  }}
#endif
};
} // namespace
int main(int argc, char** argv) {
  if (argc < 2 || !cases.count(argv[1])) return 2;
  if (argc > 2) example_package = argv[2];
  try { cases.at(argv[1])(); }
  catch (const std::exception& e) { std::cerr << argv[1] << ": " << e.what() << '\n'; return 1; }
}
