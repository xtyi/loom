#ifndef LOOM_EXAMPLE_ELEMENTWISE_H
#define LOOM_EXAMPLE_ELEMENTWISE_H

#include "abi/experimental.h"
#include <stddef.h>

/* These examples implement scalar/1D f32 only; not a runtime limitation. */
static inline int valid_buffer(const LmExpBuffer* b, uint32_t access) {
  uint64_t span;
  if (b->struct_size < sizeof(*b) || b->dtype != LM_EXP_DT_F32 ||
      b->rank > 1 || b->layout != 0 || b->access != access) return 0;
  if (b->rank == 0) span = sizeof(float);
  else {
    uint64_t last;
    if (b->dims[0] < 0 || b->strides[0] <= 0) return 0;
    if (b->dims[0] == 0) return 1;
    if ((uint64_t)(b->dims[0] - 1) >
        (UINT64_MAX / sizeof(float) - 1) / (uint64_t)b->strides[0]) return 0;
    last = (uint64_t)(b->dims[0] - 1) * (uint64_t)b->strides[0];
    span = (last + 1) * sizeof(float);
  }
  return b->data && (uintptr_t)b->data % sizeof(float) == 0 && b->span_bytes >= span;
}

static inline int same_shape(const LmExpBuffer* a, const LmExpBuffer* b) {
  return a->rank == b->rank && (a->rank == 0 || a->dims[0] == b->dims[0]);
}

static inline int64_t count(const LmExpBuffer* b) { return b->rank ? b->dims[0] : 1; }
static inline int64_t stride(const LmExpBuffer* b) { return b->rank ? b->strides[0] : 1; }
#endif
