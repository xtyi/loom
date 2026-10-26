#include "cpu/package.h"

namespace loom::cpu {
std::unique_ptr<Session> make_session(const io::Package& package) {
  std::vector<KernelLibrary> libraries;
  for (const auto& library : package.libraries) {
    libraries.push_back({library.ref, library.path.string()});
  }
  std::map<plan::SlotId, BufferView> constants;
  for (const auto& [id, data] : package.constants) {
    constants.emplace(id, BufferView{data.data.get(), data.bytes});
  }
  return std::make_unique<Session>(package.plan.graph, package.entries, libraries,
                                   constants, package.plan.order);
}

std::unique_ptr<Session> load_package(const std::filesystem::path& directory) {
  return make_session(io::load_package(directory));
}
} // namespace loom::cpu
