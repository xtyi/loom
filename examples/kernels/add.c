#include "elementwise.h"

static const LmExpCaps caps = {
  sizeof(LmExpCaps), LM_EXP_ABI_VERSION, LM_EXP_BACKEND_CPU, LM_EXP_SYNCHRONOUS,
  "add", "cpu", "lm_ku_entry_exp_v1", "f32-add/v1",
  2, 1, 0, 0, 1, 4
};

LM_EXP_EXPORT const LmExpCaps* lm_ku_caps_exp_v1(void) { return &caps; }

LM_EXP_EXPORT int32_t lm_ku_entry_exp_v1(const LmExpKernelArgs* a, void* stream) {
  const LmExpBuffer *x, *bias, *y;
  int64_t i;
  if (!a || a->struct_size < sizeof(*a) || a->abi_version != LM_EXP_ABI_VERSION ||
      stream || a->n_in != 2 || a->n_out != 1 || !a->inputs || !a->outputs ||
      a->params_bytes || a->workspace_bytes) return 1;
  x = &a->inputs[0]; bias = &a->inputs[1]; y = &a->outputs[0];
  if (!valid_buffer(x, LM_EXP_READ) || !valid_buffer(bias, LM_EXP_READ) ||
      !valid_buffer(y, LM_EXP_WRITE) || !same_shape(x, bias) || !same_shape(x, y)) return 2;
  for (i = 0; i < count(y); ++i) {
    ((float*)y->data)[i * stride(y)] =
        ((const float*)x->data)[i * stride(x)] + ((const float*)bias->data)[i * stride(bias)];
  }
  return 0;
}
