#ifndef LOOM_NV_BACKEND_H
#define LOOM_NV_BACKEND_H
#include "execution/backend.h"
namespace loom::nv {
// 创建独占的非默认 CUDA stream；无 CUDA 设备时抛出 execution::Error。
std::unique_ptr<execution::Backend> make_backend(int device = 0);
}
#endif
