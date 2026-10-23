#ifndef LOOM_PLAN_ENTRY_H
#define LOOM_PLAN_ENTRY_H

#include "plan/tensor.h"

namespace loom::plan {

// S3: in-memory entry metadata, not a callable function pointer or dlopen ABI.
struct EntryRef {
  std::string kernel; // KernelUnit 名称，例如 cast
  std::string implementation; // 具体实现名称，例如 cpu
};

struct PortContract {
  std::string name;
  TensorDesc tensor;  // Exact dtype/shape/normalized strides/layout; min alignment.
};

struct EntryContract {
  EntryRef ref;
  std::string symbol;
  std::vector<PortContract> inputs;   // Read only, positional.
  std::vector<PortContract> outputs;  // Write only, positional.
  std::string numerical_contract;   // Required opaque version/content id.
  std::size_t params_bytes = 0;      // Opaque bytes, no inferred schema/conversion.
  std::size_t workspace_bytes = 0;
  std::size_t workspace_alignment = 1;
};

struct EntryBinding {
  EntryRef ref;
  std::string numerical_contract;  // Required contract identity for this call.
  std::vector<std::uint8_t> params;
};

}  // namespace loom::plan

#endif
