#ifndef LOOM_CPU_SESSION_H
#define LOOM_CPU_SESSION_H

#include "plan/program.h"

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace loom::cpu {

// S3 宿主 API；对应 abi/experimental.h 的实验 CPU 同步执行路径。
struct KernelLibrary {
  plan::EntryRef ref;
  std::string path;
};

struct BufferView {
  const void* data = nullptr;
  std::size_t bytes = 0;
};

struct OutputTensor {
  plan::SlotId slot;
  plan::TensorInfo info;
  std::shared_ptr<const void> data; // 拥有结果；Session 销毁或再次运行后仍有效。
};

struct RunResult {
  std::vector<OutputTensor> outputs; // 与 graph.outputs 顺序一致，允许重复。
};

enum class ErrorCode {
  invalid_program, invalid_library, load, symbol, abi, unsupported,
  contract, binding, capacity, alignment, alias, kernel, kernel_protocol
};

class Error : public std::runtime_error {
 public:
  Error(ErrorCode code, std::string message,
        std::optional<plan::NodeId> node = {},
        std::optional<plan::SlotId> slot = {}, std::int32_t kernel_status = 0);
  ErrorCode code;
  std::optional<plan::NodeId> node;
  std::optional<plan::SlotId> slot;
  std::int32_t kernel_status;
  std::optional<plan::Diagnostic> diagnostic; // prepare_program 的原始诊断。
};

class Session {
 public:
  // 校验并保存独立的 PreparedProgram，按 ref 加载使用到的动态库。
  // constants 必须恰好覆盖不在 graph.inputs 中的 external slot；创建时拷贝。
  // Library 映射不允许重复或未使用项。所有权通过 RAII 管理。
  Session(const plan::Graph& graph, const std::vector<plan::EntryContract>& entries,
          const std::vector<KernelLibrary>& libraries,
          const std::map<plan::SlotId, BufferView>& constants = {},
          const std::optional<std::vector<plan::NodeId>>& order = std::nullopt);
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  // inputs 按 graph.inputs 顺序绑定，借用到同步 run 返回；调用期间不可修改。
  // 地址需满足 slot 对齐、容量覆盖存储跨度，不同输入 slot 的跨度不得重叠。
  // 先检查所有输入并分配内存，再执行；kernel 失败立即停止，不返回部分结果。
  // 每次 run 分配独立中间 buffer；不承诺同一 Session 的并发调用安全。
  RunResult run(const std::vector<BufferView>& inputs);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace loom::cpu
#endif
