#include "nv/session.h"
#include "nv/backend.h"
namespace loom::nv {
Session::Session(const plan::Graph& graph,
                 const std::vector<plan::EntryContract>& entries,
                 const std::vector<KernelLibrary>& libraries,
                 const std::map<plan::SlotId, BufferView>& constants,
                 const std::optional<std::vector<plan::NodeId>>& order, int device)
    : execution::Session(make_backend(device), graph, entries, libraries, constants, order) {}
} // namespace loom::nv
