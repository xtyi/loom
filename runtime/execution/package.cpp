#include "execution/package.h"
#include "execution/backend.h"
namespace loom::execution {
PackageBindings::PackageBindings(const io::Package& package) {
  for (const auto& library : package.libraries) {
    libraries.push_back({library.ref, library.path.string()});
  }
  for (const auto& [id, data] : package.constants) {
    constants.emplace(id, BufferView{data.data.get(), data.bytes});
  }
}
void check_package_target(const io::Package& package, const std::string& backend, const std::string& arch) {
  const auto wanted = package.backend == "cuda" ? "sm_" + std::to_string(package.compute_capability) : "";
  if (package.backend != backend || wanted != arch || (backend == "cpu" && package.compute_capability != 0)) {
    throw Error(ErrorCode::unsupported, "package target " + package.backend +
        "/" + wanted + " does not match backend " +
        backend + "/" + arch);
  }
}
std::unique_ptr<Session> make_session(const io::Package& package, std::unique_ptr<Backend> backend) {
  if (!backend) throw Error(ErrorCode::unsupported, "null backend");
  check_package_target(package, backend->name(), backend->device_arch());
  const PackageBindings bindings(package);
  return std::make_unique<Session>(std::move(backend), package.plan.graph, package.entries,
      bindings.libraries, bindings.constants, package.plan.order);
}
} // namespace loom::execution
