#include "abi/experimental.h"

/* Faults exercise the actual shared-object boundary, not a mocked resolver. */
static LmExpCaps caps = {
  sizeof(LmExpCaps), LM_EXP_ABI_VERSION, LM_EXP_BACKEND_CPU, LM_EXP_SYNCHRONOUS,
  "add", "cpu", "lm_ku_entry_exp_v1", "f32-add/v1", 2, 1, 0, 0, 1, 4
};
static int calls;
LM_EXP_EXPORT int lm_test_calls(void) { return calls; }

#if FIXTURE_MODE != 12
LM_EXP_EXPORT const LmExpCaps* lm_ku_caps_exp_v1(void) {
  switch (FIXTURE_MODE) {
    case 1: caps.abi_version = 999; break;
    case 2: caps.struct_size = sizeof(uint32_t); break;
    case 3: caps.backend = 999; break;
    case 4: caps.flags = 0; break;
    case 5: caps.implementation = "wrong"; break;
    case 6: caps.numerical_contract = "wrong"; break;
    case 7: caps.min_buffer_alignment = 3; break;
    case 8: caps.n_in = 1; break;
    case 9: caps.workspace_bytes = 16; break;
    case 10: caps.params_bytes = 1; break;
    case 13: return 0;
    case 16: caps.flags |= 0x80000000u; break;
    case 17: caps.min_buffer_alignment = 32; break;
    default: break;
  }
  return &caps;
}
#endif

#if FIXTURE_MODE != 11
LM_EXP_EXPORT int32_t lm_ku_entry_exp_v1(const LmExpKernelArgs* args, void* stream) {
  (void)args; (void)stream;
  ++calls;
  /* Keep caps used even in the missing-caps fixture. */
  return FIXTURE_MODE == 14 ? -7 : (int32_t)caps.n_out + 36;
}
#endif
