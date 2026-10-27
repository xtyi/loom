#include "abi/experimental.h"
#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <thread>

namespace {
const LmExpCaps caps = {
  sizeof(LmExpCaps), LM_EXP_ABI_VERSION, LM_EXP_BACKEND_CUDA, LM_EXP_STREAM_ORDERED,
  "add", "cuda", "lm_ku_entry_exp_v1", "f32-add/v1", 2, 1, 0, 0, 1, 4
};
int calls = 0, status = 37;
std::atomic<bool> completed{false};
void CUDART_CB finish(void*) {
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  completed.store(true);
}
__global__ void write_output(float* out) { *out = 42.f; }
}
extern "C" LM_EXP_EXPORT const LmExpCaps* lm_ku_caps_exp_v1() { return &caps; }
extern "C" LM_EXP_EXPORT int lm_test_calls() { return calls; }
extern "C" LM_EXP_EXPORT int lm_test_completed() { return completed.load(); }
extern "C" LM_EXP_EXPORT void lm_test_status(int value) { status = value; }
extern "C" LM_EXP_EXPORT int32_t lm_ku_entry_exp_v1(const LmExpKernelArgs* args, void* stream) {
  ++calls;
  completed.store(false);
  // status == 0 故意触发设备执行错误，验证错误归属当前 node。
  write_output<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
      status == 0 ? nullptr : static_cast<float*>(args->outputs[0].data));
  if (cudaGetLastError() != cudaSuccess) return 99;
  if (status != 0 && cudaLaunchHostFunc(static_cast<cudaStream_t>(stream), finish, nullptr) != cudaSuccess) return 99;
  return status;
}
