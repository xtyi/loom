#include "plan/program.h"

#include <limits>
#include <set>
#include <sstream>
#include <utility>

namespace loom::plan {
namespace {

using EntryKey = std::pair<std::string, std::string>;

EntryKey key(const EntryRef& ref) { return {ref.kernel, ref.implementation}; }

bool valid_name(const std::string& name) {
  return !name.empty() && name.find('\0') == std::string::npos;
}

Diagnostic diagnostic(ErrorCode code, std::string message, std::string field,
                      std::string expected = {}, std::string actual = {},
                      std::optional<NodeId> node = std::nullopt,
                      std::optional<SlotId> slot = std::nullopt) {
  return {code, std::move(message), node, slot, std::move(field),
          std::move(expected), std::move(actual)};
}

std::string dimensions(const std::vector<std::int64_t>& values) {
  std::ostringstream out;
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out << ',';
    out << values[i];
  }
  out << ']';
  return out.str();
}

std::optional<Diagnostic> validate_entry(EntryContract& entry) {
  const auto prefix = "entries[" + entry.ref.kernel + "/" +
                      entry.ref.implementation + "].";
  if (!valid_name(entry.ref.kernel) || !valid_name(entry.ref.implementation) ||
      !valid_name(entry.symbol) || !valid_name(entry.numerical_contract)) {
    return diagnostic(ErrorCode::invalid_entry,
                      "entry identity, symbol and numerical contract are required",
                      prefix + "identity", "nonempty strings without NUL");
  }
  if (!valid_alignment(entry.workspace_alignment)) {
    return diagnostic(ErrorCode::invalid_entry, "invalid workspace alignment",
                      prefix + "workspace_alignment", "nonzero power of two",
                      std::to_string(entry.workspace_alignment));
  }
  const auto max_bytes = static_cast<std::uint64_t>(
      std::numeric_limits<std::int64_t>::max());
  if (entry.params_bytes > max_bytes || entry.workspace_bytes > max_bytes) {
    return diagnostic(ErrorCode::invalid_entry,
                      "parameter/workspace size exceeds signed 64-bit extent",
                      prefix + "bytes");
  }
  const auto ports = [&](std::vector<PortContract>& contracts,
                         const std::string& direction) -> std::optional<Diagnostic> {
    std::set<std::string> names;
    for (std::size_t i = 0; i < contracts.size(); ++i) {
      auto& port = contracts[i];
      const auto field = prefix + direction + "[" + std::to_string(i) + "].";
      if (!valid_name(port.name) || !names.insert(port.name).second) {
        return diagnostic(ErrorCode::invalid_entry, "invalid or duplicate port name",
                          field + "name", "unique nonempty name without NUL",
                          port.name);
      }
      auto result = validate_tensor(port.tensor);
      if (!result.ok()) {
        return diagnostic(ErrorCode::invalid_entry, result.error->message,
                          field + result.error->field);
      }
      port.tensor = std::move(result.value->desc);
    }
    return std::nullopt;
  };
  if (auto error = ports(entry.inputs, "inputs")) return error;
  return ports(entry.outputs, "outputs");
}

std::optional<Diagnostic> match_port(const TensorDesc& actual,
                                     const TensorDesc& expected,
                                     NodeId node, SlotId slot,
                                     const std::string& prefix) {
  const auto mismatch = [&](const std::string& field, const std::string& want,
                            const std::string& have) {
    return diagnostic(ErrorCode::contract_mismatch,
                      "tensor does not satisfy positional entry contract",
                      prefix + field, want, have, node, slot);
  };
  if (actual.dtype != expected.dtype) {
    return mismatch("dtype", dtype_name(expected.dtype), dtype_name(actual.dtype));
  }
  if (actual.shape != expected.shape) {
    return mismatch("shape", dimensions(expected.shape), dimensions(actual.shape));
  }
  if (actual.layout != expected.layout) {
    return mismatch("layout", std::to_string(expected.layout),
                    std::to_string(actual.layout));
  }
  if (actual.strides != expected.strides) {
    return mismatch("strides", dimensions(expected.strides),
                    dimensions(actual.strides));
  }
  if (actual.alignment < expected.alignment) {
    return mismatch("alignment", ">=" + std::to_string(expected.alignment),
                    std::to_string(actual.alignment));
  }
  return std::nullopt;
}

}  // namespace

ProgramResult prepare_program(
    const Graph& graph, const std::vector<EntryContract>& entries,
    const std::optional<std::vector<NodeId>>& explicit_order) {
  const auto topology = prepare(graph, explicit_order);
  if (!topology.ok()) return {std::nullopt, topology.error};

  PreparedProgram program;
  for (const auto& slot : graph.slots) {
    auto tensor = validate_tensor(slot.tensor);
    if (!tensor.ok()) {
      return {std::nullopt,
              diagnostic(ErrorCode::invalid_tensor, tensor.error->message,
                         tensor.error->field, {}, {}, {}, slot.id)};
    }
    program.tensors.emplace(slot.id,
                            PreparedTensor{slot.external, std::move(*tensor.value)});
  }

  std::map<EntryKey, EntryContract> registry;
  for (auto entry : entries) {
    if (auto error = validate_entry(entry)) return {std::nullopt, std::move(error)};
    const auto entry_key = key(entry.ref);
    if (!registry.emplace(entry_key, std::move(entry)).second) {
      return {std::nullopt,
              diagnostic(ErrorCode::duplicate_entry, "duplicate kernel/implementation",
                         "entry", "unique kernel/implementation",
                         entry_key.first + "/" + entry_key.second)};
    }
  }
  std::map<NodeId, const Node*> nodes;
  for (const auto& node : graph.nodes) nodes.emplace(node.id, &node);
  for (const auto id : topology.order) {
    const auto& node = *nodes.at(id);
    const auto& binding = node.binding;
    const auto found = registry.find(key(binding.ref));
    if (found == registry.end()) {
      return {std::nullopt,
              diagnostic(ErrorCode::unknown_entry, "node has no registered entry",
                         "binding", "registered kernel/implementation",
                         binding.ref.kernel + "/" + binding.ref.implementation, id)};
    }
    const auto& entry = found->second;
    if (binding.numerical_contract != entry.numerical_contract) {
      return {std::nullopt,
              diagnostic(ErrorCode::contract_mismatch, "numerical contract mismatch",
                         "numerical_contract", entry.numerical_contract,
                         binding.numerical_contract, id)};
    }
    if (binding.params.size() != entry.params_bytes) {
      return {std::nullopt,
              diagnostic(ErrorCode::contract_mismatch, "parameter byte count mismatch",
                         "params_bytes", std::to_string(entry.params_bytes),
                         std::to_string(binding.params.size()), id)};
    }
    const auto match = [&](const std::vector<SlotId>& slots,
                           const std::vector<PortContract>& ports,
                           const std::string& direction) -> std::optional<Diagnostic> {
      if (slots.size() != ports.size()) {
        return diagnostic(ErrorCode::contract_mismatch, "port count mismatch",
                          direction + ".count", std::to_string(ports.size()),
                          std::to_string(slots.size()), id);
      }
      for (std::size_t i = 0; i < slots.size(); ++i) {
        if (auto error = match_port(program.tensors.at(slots[i]).info.desc,
                                    ports[i].tensor, id, slots[i],
                                    direction + "[" + std::to_string(i) + "].")) {
          return error;
        }
      }
      return std::nullopt;
    };
    if (auto error = match(node.inputs, entry.inputs, "inputs")) {
      return {std::nullopt, std::move(error)};
    }
    if (auto error = match(node.outputs, entry.outputs, "outputs")) {
      return {std::nullopt, std::move(error)};
    }
    program.calls.push_back({id, entry, node.inputs, node.outputs, binding.params});
  }
  program.inputs = graph.inputs;
  program.outputs = graph.outputs;
  return {std::move(program), std::nullopt};
}

}  // namespace loom::plan
