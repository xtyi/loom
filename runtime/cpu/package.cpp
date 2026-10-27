#include "cpu/package.h"
#include "execution/package.h"

namespace loom::cpu {
std::unique_ptr<Session> make_session(const io::Package& package) {
  execution::check_package_target(package, "cpu", "");
  const execution::PackageBindings bindings(package);
  return std::make_unique<Session>(package.plan.graph, package.entries, bindings.libraries,
                                   bindings.constants, package.plan.order);
}

std::unique_ptr<Session> load_package(const std::filesystem::path& directory) {
  return make_session(io::load_package(directory));
}
} // namespace loom::cpu
