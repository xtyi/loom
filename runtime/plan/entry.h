#ifndef LOOM_PLAN_ENTRY_H
#define LOOM_PLAN_ENTRY_H

#include "plan/tensor.h"

// EntryContract 描述实现的接口，EntryBinding 描述节点的调用，TensorDesc 描述实际数据；prepare_program() 负责确认三者彼此一致
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

//可以看成一个 kernel 的"函数声明 + 资源声明"
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

// EntryBinding不是在描述"实现本身",而是在描述"图中的这一次调用",会绑定到图节点
struct EntryBinding {
  EntryRef ref;
  std::string numerical_contract;  // Required contract identity for this call.
  std::vector<std::uint8_t> params;
};

}  // namespace loom::plan

#endif
