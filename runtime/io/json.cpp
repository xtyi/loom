#include "io/json_internal.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <set>

namespace loom::io {

Error::Error(ErrorCode c, std::string s, std::string f, std::string message)
    : std::runtime_error(s + ": " + f + ": " + message), code(c),
      source(std::move(s)), field(std::move(f)) {}

namespace detail {
namespace {
constexpr std::size_t max_json_bytes = 16 * 1024 * 1024;
constexpr auto max_extent = static_cast<std::uint64_t>(
    std::numeric_limits<std::int64_t>::max());

std::size_t size(const Reader& r, const Json& j, const std::string& field) {
  return static_cast<std::size_t>(r.uint(j, field,
      std::min<std::uint64_t>(max_extent, std::numeric_limits<std::size_t>::max())));
}
} // namespace

[[noreturn]] void Reader::fail(ErrorCode code, const std::string& field,
                               const std::string& message) const {
  throw Error(code, source, field, message);
}

void Reader::object(const Json& j, const std::string& field,
                    std::initializer_list<const char*> required,
                    std::initializer_list<const char*> optional) const {
  if (!j.is_object()) fail(ErrorCode::schema, field, "expected object");
  for (const auto* key : required) {
    if (!j.contains(key)) fail(ErrorCode::schema, field + "." + key, "required field missing");
  }
  for (auto it = j.begin(); it != j.end(); ++it) {
    const auto known = [&](const auto& keys) {
      return std::any_of(keys.begin(), keys.end(), [&](const char* key) { return it.key() == key; });
    };
    if (!known(required) && !known(optional)) {
      fail(ErrorCode::schema, field + "." + it.key(), "unknown/unsupported field");
    }
  }
}

const Json& Reader::array(const Json& j, const std::string& field) const {
  if (!j.is_array()) fail(ErrorCode::schema, field, "expected array");
  return j;
}

std::string Reader::string(const Json& j, const std::string& field) const {
  if (!j.is_string()) fail(ErrorCode::schema, field, "expected string");
  auto s = j.get<std::string>();
  if (s.empty() || s.find('\0') != std::string::npos) {
    fail(ErrorCode::schema, field, "expected nonempty string without NUL");
  }
  return s;
}

std::uint64_t Reader::uint(const Json& j, const std::string& field, std::uint64_t max) const {
  if (!j.is_number_integer() ||
      (!j.is_number_unsigned() && j.get<std::int64_t>() < 0)) {
    fail(ErrorCode::schema, field, "expected nonnegative integer (no float or boolean)");
  }
  const auto value = j.get<std::uint64_t>();
  if (value > max) fail(ErrorCode::schema, field, "integer out of range");
  return value;
}

bool Reader::boolean(const Json& j, const std::string& field) const {
  if (!j.is_boolean()) fail(ErrorCode::schema, field, "expected boolean");
  return j.get<bool>();
}

void Reader::header(const Json& j, const char* format,
                    std::initializer_list<const char*> capabilities) const {
  if (!j.is_object() || !j.contains("format")) {
    fail(ErrorCode::schema, "$.format", "required format missing");
  }
  if (string(j.at("format"), "$.format") != format) {
    fail(ErrorCode::version, "$.format", std::string("requires exactly ") + format);
  }
  if (j.contains("required_capabilities")) {
    const auto& caps = array(j.at("required_capabilities"), "$.required_capabilities");
    for (std::size_t i = 0; i < caps.size(); ++i) {
      const auto field = "$.required_capabilities[" + std::to_string(i) + "]";
      const auto cap = string(caps[i], field);
      if (std::none_of(capabilities.begin(), capabilities.end(),
                       [&](const char* supported) { return cap == supported; })) {
        fail(ErrorCode::capability, field, "unsupported required capability: " + cap);
      }
    }
  }
  if (j.contains("meta") && !j.at("meta").is_object()) {
    fail(ErrorCode::schema, "$.meta", "expected object");
  }
}

Json parse(std::string_view text, const Reader& r) {
  if (text.size() > max_json_bytes) r.fail(ErrorCode::syntax, "$", "JSON exceeds 16 MiB limit");
  std::vector<std::set<std::string>> keys;
  const auto callback = [&](int depth, Json::parse_event_t event, Json& value) {
    if (depth > 64) r.fail(ErrorCode::syntax, "$", "JSON nesting exceeds 64 levels");
    if (event == Json::parse_event_t::object_start) keys.emplace_back();
    if (event == Json::parse_event_t::key && !keys.back().insert(value.get<std::string>()).second) {
      r.fail(ErrorCode::syntax, "$", "duplicate JSON key: " + value.get<std::string>());
    }
    if (event == Json::parse_event_t::object_end) keys.pop_back();
    return true;
  };
  try {
    return Json::parse(text.begin(), text.end(), callback);
  } catch (const Json::exception& e) {
    r.fail(ErrorCode::syntax, "$", e.what());
  }
}

std::string read_text(const std::filesystem::path& path) {
  const Reader r{path.string()};
  std::error_code ec;
  const auto bytes = std::filesystem::file_size(path, ec);
  if (ec) r.fail(ErrorCode::io, "$", "cannot stat JSON file: " + ec.message());
  if (bytes > max_json_bytes) r.fail(ErrorCode::syntax, "$", "JSON exceeds 16 MiB limit");
  std::ifstream in(path, std::ios::binary);
  std::string text(static_cast<std::size_t>(bytes), '\0');
  if (!in || !in.read(text.data(), static_cast<std::streamsize>(text.size())) ||
      in.peek() != std::char_traits<char>::eof()) {
    r.fail(ErrorCode::io, "$", "cannot read complete JSON file");
  }
  return text;
}

std::vector<std::uint32_t> Reader::ids(const Json& j, const std::string& field) const {
  array(j, field);
  std::vector<std::uint32_t> out;
  for (std::size_t i = 0; i < j.size(); ++i) {
    out.push_back(static_cast<std::uint32_t>(uint(j[i], field + "[" + std::to_string(i) + "]",
                                                UINT32_MAX)));
  }
  return out;
}

plan::EntryRef Reader::ref(const Json& j, const std::string& field) const {
  return {string(j.at("kernel"), field + ".kernel"),
          string(j.at("implementation"), field + ".implementation")};
}

plan::TensorDesc Reader::tensor(const Json& j, const std::string& field) const {
  object(j, field, {"dtype", "shape"}, {"strides", "layout", "alignment"});
  plan::TensorDesc out;
  const auto name = string(j.at("dtype"), field + ".dtype");
  for (auto dtype : {plan::DType::f16, plan::DType::bf16, plan::DType::f32, plan::DType::f64,
                     plan::DType::i8, plan::DType::u8, plan::DType::i32, plan::DType::i64,
                     plan::DType::boolean}) {
    if (name == plan::dtype_name(dtype)) out.dtype = dtype;
  }
  const auto dims = [&](const Json& values, const std::string& path) {
    array(values, path);
    if (values.size() > 8) fail(ErrorCode::schema, path, "rank exceeds 8");
    std::vector<std::int64_t> result;
    for (std::size_t i = 0; i < values.size(); ++i) {
      result.push_back(static_cast<std::int64_t>(uint(values[i], path + "[" + std::to_string(i) + "]",
                                                     max_extent)));
    }
    return result;
  };
  out.shape = dims(j.at("shape"), field + ".shape");
  if (j.contains("strides")) out.strides = dims(j.at("strides"), field + ".strides");
  if (j.contains("layout")) out.layout = static_cast<std::uint32_t>(uint(j.at("layout"), field + ".layout", UINT32_MAX));
  if (j.contains("alignment")) out.alignment = size(*this, j.at("alignment"), field + ".alignment");
  auto validated = plan::validate_tensor(out);
  if (!validated.ok()) fail(ErrorCode::schema, field + "." + validated.error->field, validated.error->message);
  return validated.value->desc;
}

void program_error(const Reader& r, const plan::Diagnostic& diagnostic) {
  std::string context;
  if (diagnostic.node) context += " node=" + std::to_string(*diagnostic.node);
  if (diagnostic.slot) context += " slot=" + std::to_string(*diagnostic.slot);
  if (!diagnostic.expected.empty()) context += " expected=" + diagnostic.expected;
  if (!diagnostic.actual.empty()) context += " actual=" + diagnostic.actual;
  Error error(ErrorCode::program, r.source, diagnostic.field.empty() ? "$" : "$." + diagnostic.field,
              diagnostic.message + context);
  error.diagnostic = diagnostic;
  throw error;
}

plan::EntryContract entry(const Json& j, const Reader& r) {
  r.header(j, kernel_format, {"static-tensors", "cpu-sync", "cuda-stream"});
  r.object(j, "$", {"format", "kernel", "implementation", "symbol", "inputs", "outputs",
                     "numerical_contract", "library"},
                    {"params_bytes", "workspace_bytes", "workspace_alignment", "required_capabilities", "meta"});
  plan::EntryContract out;
  out.ref = r.ref(j, "$");
  out.symbol = r.string(j.at("symbol"), "$.symbol");
  out.numerical_contract = r.string(j.at("numerical_contract"), "$.numerical_contract");
  const auto ports = [&](const char* direction) {
    const std::string path = std::string("$.") + direction;
    const auto& values = r.array(j.at(direction), path);
    std::vector<plan::PortContract> result;
    for (std::size_t i = 0; i < values.size(); ++i) {
      const auto field = path + "[" + std::to_string(i) + "]";
      r.object(values[i], field, {"name", "tensor"});
      result.push_back({r.string(values[i].at("name"), field + ".name"),
                        r.tensor(values[i].at("tensor"), field + ".tensor")});
    }
    return result;
  };
  out.inputs = ports("inputs");
  out.outputs = ports("outputs");
  if (j.contains("params_bytes")) out.params_bytes = size(r, j.at("params_bytes"), "$.params_bytes");
  if (j.contains("workspace_bytes")) out.workspace_bytes = size(r, j.at("workspace_bytes"), "$.workspace_bytes");
  if (j.contains("workspace_alignment")) out.workspace_alignment = size(r, j.at("workspace_alignment"), "$.workspace_alignment");
  const auto checked = plan::prepare_program(plan::Graph{}, {out});
  if (!checked.ok()) program_error(r, *checked.error);
  return out;
}
} // namespace detail

PlanDocument parse_plan(std::string_view text, const std::string& source) {
  const detail::Reader r{source};
  const auto j = detail::parse(text, r);
  r.header(j, plan_format, {"static-tensors", "single-stream"});
  r.object(j, "$", {"format", "slots", "nodes", "inputs", "outputs"},
                    {"schedule", "required_capabilities", "meta"});
  PlanDocument out;
  const auto& slots = r.array(j.at("slots"), "$.slots");
  for (std::size_t i = 0; i < slots.size(); ++i) {
    const auto field = "$.slots[" + std::to_string(i) + "]";
    const auto& slot = slots[i];
    r.object(slot, field, {"id", "external", "tensor"});
    // Persisted IDs must survive file ordering and sparse IDs; do not auto-allocate.
    out.graph.slots.push_back({static_cast<plan::SlotId>(r.uint(slot.at("id"), field + ".id", UINT32_MAX)),
                              r.boolean(slot.at("external"), field + ".external"),
                              r.tensor(slot.at("tensor"), field + ".tensor")});
  }
  const auto& nodes = r.array(j.at("nodes"), "$.nodes");
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const auto field = "$.nodes[" + std::to_string(i) + "]";
    const auto& node = nodes[i];
    r.object(node, field, {"id", "inputs", "outputs", "binding"}, {"after"});
    plan::Node n;
    n.id = static_cast<plan::NodeId>(r.uint(node.at("id"), field + ".id", UINT32_MAX));
    n.inputs = r.ids(node.at("inputs"), field + ".inputs");
    n.outputs = r.ids(node.at("outputs"), field + ".outputs");
    if (node.contains("after")) n.after = r.ids(node.at("after"), field + ".after");
    const auto& binding = node.at("binding");
    const auto bf = field + ".binding";
    r.object(binding, bf, {"kernel", "implementation", "numerical_contract"}, {"params_hex"});
    n.binding.ref = r.ref(binding, bf);
    n.binding.numerical_contract = r.string(binding.at("numerical_contract"), bf + ".numerical_contract");
    if (binding.contains("params_hex")) {
      const auto& value = binding.at("params_hex");
      if (!value.is_string()) r.fail(ErrorCode::schema, bf + ".params_hex", "expected hex string");
      const auto hex = value.get<std::string>();
      const auto digit = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
      };
      if (hex.size() % 2) r.fail(ErrorCode::schema, bf + ".params_hex", "expected pairs of hex digits");
      for (std::size_t k = 0; k < hex.size(); k += 2) {
        if (digit(hex[k]) < 0 || digit(hex[k + 1]) < 0) r.fail(ErrorCode::schema, bf + ".params_hex", "invalid hex digit");
        n.binding.params.push_back(static_cast<std::uint8_t>(digit(hex[k]) * 16 + digit(hex[k + 1])));
      }
    }
    out.graph.nodes.push_back(std::move(n));
  }
  out.graph.inputs = r.ids(j.at("inputs"), "$.inputs");
  out.graph.outputs = r.ids(j.at("outputs"), "$.outputs");
  if (j.contains("schedule")) {
    r.object(j.at("schedule"), "$.schedule", {"order"});
    out.order = r.ids(j.at("schedule").at("order"), "$.schedule.order");
  }
  const auto result = plan::prepare(out.graph, out.order);
  if (!result.ok()) detail::program_error(r, *result.error);
  return out;
}

PlanDocument load_plan(const std::filesystem::path& path) {
  return parse_plan(detail::read_text(path), path.string());
}
} // namespace loom::io
