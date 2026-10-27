#ifndef LOOM_CPU_BACKEND_H
#define LOOM_CPU_BACKEND_H
#include "execution/backend.h"
namespace loom::cpu {
std::unique_ptr<execution::Backend> make_backend();
}
#endif
