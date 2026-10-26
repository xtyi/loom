#ifndef LOOM_CPU_PACKAGE_H
#define LOOM_CPU_PACKAGE_H

#include "cpu/session.h"
#include "io/package.h"

namespace loom::cpu {
// Session revalidates metadata and binary caps and copies constants as usual.
std::unique_ptr<Session> make_session(const io::Package& package);
std::unique_ptr<Session> load_package(const std::filesystem::path& directory);
} // namespace loom::cpu
#endif
