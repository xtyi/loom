#ifndef LOOM_CPU_SESSION_H
#define LOOM_CPU_SESSION_H
#include "execution/session.h"
namespace loom::cpu {
using execution::KernelLibrary;
using execution::BufferView;
using execution::OutputTensor;
using execution::RunResult;
using execution::ErrorCode;
using execution::Error;
// 保留 CPU 宿主 API；执行与校验由公共 Session 实现。
class Session : public execution::Session {
 public:
  Session(const plan::Graph& graph, const std::vector<plan::EntryContract>& entries,
          const std::vector<KernelLibrary>& libraries,
          const std::map<plan::SlotId, BufferView>& constants = {},
          const std::optional<std::vector<plan::NodeId>>& order = std::nullopt);
};
} // namespace loom::cpu
#endif
