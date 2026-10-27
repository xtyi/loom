#ifndef LOOM_NV_PACKAGE_H
#define LOOM_NV_PACKAGE_H
#include "nv/session.h"
#include "io/package.h"
namespace loom::nv {
std::unique_ptr<execution::Session> make_session(const io::Package& package, int device = 0);
std::unique_ptr<execution::Session> load_package(const std::filesystem::path& directory, int device = 0);
}
#endif
