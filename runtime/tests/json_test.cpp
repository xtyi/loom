#include "io/json.h"
#include <nlohmann/json.hpp>

#include <functional>
#include <iostream>
#include <map>

using nlohmann::json;
using namespace loom::io;
namespace {
void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }

json document() {
  return json::parse(R"({
    "format":"loom-plan-exp/1",
    "slots":[
      {"id":100,"external":true,"tensor":{"dtype":"f32","shape":[2,3],"alignment":16}},
      {"id":7,"external":false,"tensor":{"dtype":"f16","shape":[2,3],"alignment":16}},
      {"id":42,"external":false,"tensor":{"dtype":"f16","shape":[2,3],"alignment":16}}
    ],
    "nodes":[
      {"id":10,"inputs":[7],"outputs":[42],"binding":{"kernel":"compute","implementation":"cpu","numerical_contract":"f16-f32acc/v1","params_hex":"00000040"}},
      {"id":30,"inputs":[100],"outputs":[7],"binding":{"kernel":"cast","implementation":"cpu","numerical_contract":"rne/v1"}}
    ],"inputs":[100],"outputs":[42]
  })");
}
Error rejects(ErrorCode code, const std::function<void()>& fn, const std::string& field = {}) {
  try { fn(); } catch (const Error& e) {
    require(e.code == code, std::string("wrong error: ") + e.what());
    require(!e.source.empty() && !e.field.empty(), "missing JSON diagnostic context");
    if (!field.empty()) require(e.field == field, "wrong field: " + e.field);
    return e;
  }
  throw std::runtime_error("invalid JSON accepted");
}
void bad(const json& j, ErrorCode code = ErrorCode::schema, const std::string& field = {}) {
  rejects(code, [&] { parse_plan(j.dump(), "test.json"); }, field);
}

const std::map<std::string, std::function<void()>> cases{
  {"decode", [] {
    const auto plan = parse_plan(document().dump());
    require(plan.graph.slots[0].id == 100 && plan.graph.slots[1].id == 7, "serialized IDs renumbered");
    require(plan.graph.inputs == std::vector<std::uint32_t>{100} && plan.graph.outputs == std::vector<std::uint32_t>{42}, "interface order lost");
    require(plan.graph.slots[0].tensor.strides == std::vector<std::int64_t>({3,1}), "default stride not normalized");
    require(plan.graph.nodes[0].binding.params == std::vector<std::uint8_t>({0,0,0,64}), "parameter bytes changed");
    require(!plan.order && loom::plan::prepare(plan.graph).order == std::vector<std::uint32_t>({30,10}), "wrong topology");
    auto graph = plan.graph;
    require(graph.add_slot() == 101, "loaded graph cannot use automatic slot allocation");
  }},
  {"syntax", [] {
    for (const auto* text : {"", "{", "{} trailing", "{\"format\":1,}", "/*comment*/{}", "{\"x\":NaN}"}) {
      rejects(ErrorCode::syntax, [&] { parse_plan(text); });
    }
    auto s = document().dump();
    s.insert(1, "\"format\":\"ignored\",");
    rejects(ErrorCode::syntax, [&] { parse_plan(s); });
    auto j = document(); j["meta"] = {{"a", 1}};
    s = j.dump(); const auto at = s.find("\"a\":1"); s.insert(at, "\"a\":2,");
    rejects(ErrorCode::syntax, [&] { parse_plan(s); });
    rejects(ErrorCode::syntax, [] { parse_plan(std::string(70, '[') + "0" + std::string(70, ']')); });
    rejects(ErrorCode::syntax, [] { parse_plan(std::string(16 * 1024 * 1024 + 1, ' ')); });
  }},
  {"schema", [] {
    for (const auto* field : {"slots", "nodes", "inputs", "outputs"}) { auto j = document(); j.erase(field); bad(j); }
    for (const auto* field : {"symbols", "variants", "stream", "alias_of", "mem_plan"}) { auto j = document(); j[field] = nullptr; bad(j); }
    auto j = document(); j["nodes"][0]["stream"] = 1; bad(j);
    j = document(); j["slots"][0]["external"] = 1; bad(j);
    j = document(); j["nodes"][0]["binding"]["extra"] = true; bad(j);
    j = document(); j["meta"] = 1; bad(j);
    for (const auto& value : {json(nullptr), json::array(), json(3)}) bad(value);
  }},
  {"versions", [] {
    auto j = document(); j.erase("format"); bad(j);
    for (const auto* version : {"loom-plan-exp/2", "loom/1.0", "1.0"}) { j = document(); j["format"] = version; bad(j, ErrorCode::version); }
    j = document(); j["required_capabilities"] = {"dynamic-shape"}; bad(j, ErrorCode::capability);
    j["required_capabilities"] = {"static-tensors", "single-stream"}; j["meta"] = {{"future", {{"anything", true}}}};
    require(!parse_plan(j.dump()).graph.nodes.empty(), "valid metadata/capabilities rejected");
  }},
  {"integers", [] {
    for (const auto& value : {json(-1), json(1.0), json(true), json("7"), json(UINT64_MAX)}) {
      auto j = document(); j["slots"][0]["id"] = value; bad(j);
    }
    auto j = document(); j["slots"][0]["tensor"]["shape"] = {18446744073709551615ull}; bad(j);
    j = document(); j["slots"][0]["tensor"]["shape"] = {2.0,3}; bad(j);
    j = document(); j["slots"][0]["tensor"]["layout"] = 4294967296ull; bad(j);
  }},
  {"tensors", [] {
    for (auto tensor : {json{{"dtype","f32"},{"shape",{-1}}},
                        json{{"dtype","f8"},{"shape",{6}}},
                        json{{"dtype","f32"},{"shape",{2,3}},{"strides",{1,1}}},
                        json{{"dtype","f32"},{"shape",{2,3}},{"strides",{3,0}}},
                        json{{"dtype","f32"},{"shape",{2,3}},{"alignment",3}},
                        json{{"dtype","f64"},{"shape",{9223372036854775807ll}}}}) {
      auto j = document(); j["slots"][0]["tensor"] = tensor; bad(j);
    }
    auto j = document(); j["slots"][0]["tensor"]["shape"] = json::array();
    require(parse_plan(j.dump()).graph.slots[0].tensor.shape.empty(), "scalar rejected");
    j["slots"][0]["tensor"]["shape"] = {0};
    require(parse_plan(j.dump()).graph.slots[0].tensor.shape[0] == 0, "empty tensor rejected");
  }},
  {"topology", [] {
    auto j = document(); j["nodes"][0]["inputs"] = {999};
    auto e = rejects(ErrorCode::program, [&] { parse_plan(j.dump()); });
    require(e.diagnostic && e.diagnostic->node == 10 && e.diagnostic->slot == 999, "lost plan diagnostic");
    j = document(); j["nodes"][1]["after"] = {10}; bad(j, ErrorCode::program);
    j = document(); j["slots"].push_back(j["slots"][0]); bad(j, ErrorCode::program);
    j = document(); j["inputs"] = {7}; bad(j, ErrorCode::program);
  }},
  {"schedule", [] {
    auto j = document(); j["schedule"] = {{"order", {30,10}}};
    require(parse_plan(j.dump()).order == std::vector<std::uint32_t>({30,10}), "explicit order lost");
    for (const auto& order : {json{10,30}, json{30,30}, json{30}, json::array()}) { j["schedule"]["order"] = order; bad(j, ErrorCode::program); }
    j["schedule"] = nullptr; bad(j);
  }},
  {"params", [] {
    for (const auto& value : {json("0"), json("0g"), json("00 40"), json(7), json(nullptr)}) {
      auto j = document(); j["nodes"][0]["binding"]["params_hex"] = value; bad(j);
    }
    auto j = document(); j["nodes"][0]["binding"]["params_hex"] = "00FFaB";
    require(parse_plan(j.dump()).graph.nodes[0].binding.params == std::vector<std::uint8_t>({0,255,171}), "hex decode failed");
    j["nodes"][0]["binding"]["params_hex"] = "";
    require(parse_plan(j.dump()).graph.nodes[0].binding.params.empty(), "empty params rejected");
    j["nodes"][0]["binding"]["numerical_contract"] = ""; bad(j);
  }}
};
} // namespace
int main(int argc, char** argv) {
  if (argc != 2 || !cases.count(argv[1])) return 2;
  try { cases.at(argv[1])(); }
  catch (const std::exception& e) { std::cerr << argv[1] << ": " << e.what() << '\n'; return 1; }
}
