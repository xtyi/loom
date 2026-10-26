#ifndef LOOM_IO_JSON_H
#define LOOM_IO_JSON_H

#include "plan/program.h"

#include <filesystem>
#include <stdexcept>
#include <string_view>

namespace loom::io {

// Experimental on-disk formats, distinct from the target v1 formats in docs/04-06.
inline constexpr char plan_format[] = "loom-plan-exp/1";
inline constexpr char package_format[] = "loom-package-exp/1";
inline constexpr char kernel_format[] = "loom-kernel-exp/1";

enum class ErrorCode { io, syntax, schema, version, capability, target, path, program, binding };

class Error : public std::runtime_error {
 public:
  Error(ErrorCode code, std::string source, std::string field, std::string message);
  ErrorCode code;
  std::string source;
  std::string field; // JSON path, e.g. $.nodes[0].binding.params_hex.
  std::optional<plan::Diagnostic> diagnostic;
};

struct PlanDocument {
  plan::Graph graph;
  std::optional<std::vector<plan::NodeId>> order;
};

// Strict JSON decoding + tensor/topology validation. Does not resolve kernels.
// Unknown execution fields and duplicate keys are rejected; only meta is opaque.
PlanDocument parse_plan(std::string_view json, const std::string& source = "<memory>");
PlanDocument load_plan(const std::filesystem::path& path);

} // namespace loom::io
#endif
