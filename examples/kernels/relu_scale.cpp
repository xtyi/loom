#include "elementwise.h"

#include <cmath>
#include <cstring>

namespace {
const LmExpCaps caps = {
  sizeof(LmExpCaps), LM_EXP_ABI_VERSION, LM_EXP_BACKEND_CPU, LM_EXP_SYNCHRONOUS,
  "relu_scale", "cpu", "lm_ku_entry_exp_v1", "f32-relu-scale/v1",
  1, 1, sizeof(float), 16, 16, 4
};
}

extern "C" LM_EXP_EXPORT const LmExpCaps* lm_ku_caps_exp_v1(void) { return &caps; }

extern "C" LM_EXP_EXPORT int32_t lm_ku_entry_exp_v1(const LmExpKernelArgs* a, void* stream) {
  if (!a || a->struct_size < sizeof(*a) || a->abi_version != LM_EXP_ABI_VERSION ||
      stream || a->n_in != 1 || a->n_out != 1 || !a->inputs || !a->outputs ||
      a->params_bytes != sizeof(float) || !a->params || a->workspace_bytes < 16 ||
      !a->workspace || reinterpret_cast<uintptr_t>(a->workspace) % 16) return 1;
  const auto& x = a->inputs[0];
  const auto& y = a->outputs[0];
  if (!valid_buffer(&x, LM_EXP_READ) || !valid_buffer(&y, LM_EXP_WRITE) ||
      !same_shape(&x, &y)) return 2;
  float factor;
  std::memcpy(&factor, a->params, sizeof(factor));
  if (!std::isfinite(factor)) return 3;
  // Exercise caller-provided scratch without requiring params alignment.
  std::memcpy(a->workspace, &factor, sizeof(factor));
  for (int64_t i = 0; i < count(&y); ++i) {
    const float value = static_cast<const float*>(x.data)[i * stride(&x)];
    float scale;
    std::memcpy(&scale, a->workspace, sizeof(scale));
    // Preserve NaN and signed zero through the comparison; no fast-math.
    static_cast<float*>(y.data)[i * stride(&y)] = (value < 0.f ? 0.f : value) * scale;
  }
  return 0;
}
