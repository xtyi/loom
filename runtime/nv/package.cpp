#include "nv/package.h"
#include "execution/package.h"
#include "nv/backend.h"
namespace loom::nv {
std::unique_ptr<execution::Session> make_session(const io::Package& package, int device) {
  return execution::make_session(package, make_backend(device));
}
std::unique_ptr<execution::Session> load_package(const std::filesystem::path& directory, int device) {
  return make_session(io::load_package(directory), device);
}
}
