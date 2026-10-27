#include "cpu/session.h"
#include "cpu/backend.h"
namespace loom::cpu {
Session::Session(const plan::Graph& graph,
                 const std::vector<plan::EntryContract>& entries,
                 const std::vector<KernelLibrary>& libraries,
                 const std::map<plan::SlotId, BufferView>& constants,
                 const std::optional<std::vector<plan::NodeId>>& order)
    : execution::Session(make_backend(), graph, entries, libraries, constants, order) {}
} // namespace loom::cpu
