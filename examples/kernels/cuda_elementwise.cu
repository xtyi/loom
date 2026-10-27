#include "elementwise.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>

namespace {
#ifdef LOOM_CUDA_add
const LmExpCaps caps = {
  sizeof(LmExpCaps), LM_EXP_ABI_VERSION, LM_EXP_BACKEND_CUDA, LM_EXP_STREAM_ORDERED,
  "add", "cuda", "lm_ku_entry_exp_v1", "f32-add/v1", 2, 1, 0, 0, 1, 4
};
__global__ void add(const float* x, const float* bias, float* y,
                    int64_t n, int64_t sx, int64_t sb, int64_t sy) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += int64_t(blockDim.x) * gridDim.x) y[i * sy] = x[i * sx] + bias[i * sb];
}
#else
const LmExpCaps caps = {
  sizeof(LmExpCaps), LM_EXP_ABI_VERSION, LM_EXP_BACKEND_CUDA, LM_EXP_STREAM_ORDERED,
  "relu_scale", "cuda", "lm_ku_entry_exp_v1", "f32-relu-scale/v1", 1, 1, sizeof(float), 16, 16, 4
};
__global__ void set_scale(float* workspace, float scale) { *workspace = scale; }
__global__ void relu(const float* x, float* y, int64_t n, int64_t sx, int64_t sy,
                     const float* workspace) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += int64_t(blockDim.x) * gridDim.x) {
    const float value = x[i * sx];
    y[i * sy] = (value < 0.f ? 0.f : value) * *workspace;
  }
}
#endif
}
extern "C" LM_EXP_EXPORT const LmExpCaps* lm_ku_caps_exp_v1(void) { return &caps; }
extern "C" LM_EXP_EXPORT int32_t lm_ku_entry_exp_v1(const LmExpKernelArgs* a, void* stream) {
  if (!a || a->struct_size < sizeof(*a) || a->abi_version != LM_EXP_ABI_VERSION ||
      !stream || a->n_in != caps.n_in || a->n_out != 1 || !a->inputs || !a->outputs ||
      a->params_bytes != caps.params_bytes || a->workspace_bytes != caps.workspace_bytes) return 1;
  const auto& x = a->inputs[0];
  const auto& y = a->outputs[0];
  if (!valid_buffer(&x, LM_EXP_READ) || !valid_buffer(&y, LM_EXP_WRITE) || !same_shape(&x, &y)) return 2;
  const auto n = count(&y);
  const auto blocks = n / 256 + (n % 256 != 0);
  const auto grid = static_cast<unsigned>(blocks > 65535 ? 65535 : blocks);
  const auto cuda_stream = static_cast<cudaStream_t>(stream);
#ifdef LOOM_CUDA_add
  const auto& bias = a->inputs[1];
  if (!valid_buffer(&bias, LM_EXP_READ) || !same_shape(&x, &bias)) return 2;
  if (n) add<<<grid, 256, 0, cuda_stream>>>(static_cast<const float*>(x.data),
      static_cast<const float*>(bias.data), static_cast<float*>(y.data),
      n, stride(&x), stride(&bias), stride(&y));
#else
  if (!a->params || !a->workspace || reinterpret_cast<uintptr_t>(a->workspace) % 16) return 1;
  float scale;
  std::memcpy(&scale, a->params, sizeof(scale));
  if (!std::isfinite(scale)) return 3;
  // 不把栈上的 params 地址交给异步拷贝；按值传参，在同一 stream 写设备 scratch。
  set_scale<<<1, 1, 0, cuda_stream>>>(static_cast<float*>(a->workspace), scale);
  if (cudaGetLastError() != cudaSuccess) return 4;
  if (n) relu<<<grid, 256, 0, cuda_stream>>>(static_cast<const float*>(x.data),
      static_cast<float*>(y.data), n, stride(&x), stride(&y), static_cast<const float*>(a->workspace));
#endif
  return cudaGetLastError() == cudaSuccess ? 0 : 4;
}
