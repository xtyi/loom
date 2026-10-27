#ifndef LOOM_EXECUTION_PACKAGE_H
#define LOOM_EXECUTION_PACKAGE_H
#include "execution/session.h"
#include "io/package.h"
namespace loom::execution {
// 包到执行器的公共适配；借用常量仅用于 Session 创建时拷贝。
struct PackageBindings {
  explicit PackageBindings(const io::Package& package);
  std::vector<KernelLibrary> libraries;
  std::map<plan::SlotId, BufferView> constants;
};
void check_package_target(const io::Package& package, const std::string& backend,
                          const std::string& device_arch);
std::unique_ptr<Session> make_session(const io::Package& package, std::unique_ptr<Backend> backend);
} // namespace loom::execution
#endif
