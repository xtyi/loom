#ifndef LOOM_NV_SESSION_H
#define LOOM_NV_SESSION_H
#include "execution/session.h"
namespace loom::nv {
using execution::KernelLibrary;
using execution::BufferView;
using execution::OutputTensor;
using execution::RunResult;
using execution::ErrorCode;
using execution::Error;
class Session : public execution::Session {
 public:
  Session(const plan::Graph& graph, const std::vector<plan::EntryContract>& entries,
          const std::vector<KernelLibrary>& libraries,
          const std::map<plan::SlotId, BufferView>& constants = {},
          const std::optional<std::vector<plan::NodeId>>& order = std::nullopt,
          int device = 0);
};
} // namespace loom::nv
#endif
