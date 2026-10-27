#ifndef LOOM_ABI_EXPERIMENTAL_H
#define LOOM_ABI_EXPERIMENTAL_H

#include <stdint.h>

/* S2 experimental backend ABI, NOT the draft lm_ku_*_v1 ABI.
 * Opt in with -DLOOM_ENABLE_EXPERIMENTAL_ABI=ON when building Loom.
 * Same native architecture/calling convention required; no packed structs.
 * Version must match exactly. Changes require rebuilding the KernelUnits. */
#define LM_EXP_ABI_VERSION 1u
#define LM_EXP_MAX_DIMS 8u
#define LM_EXP_BACKEND_CPU 1u
#define LM_EXP_BACKEND_CUDA 2u
#define LM_EXP_SYNCHRONOUS 1u
#define LM_EXP_STREAM_ORDERED 2u
#define LM_EXP_READ 1u
#define LM_EXP_WRITE 2u

#define LM_EXP_DT_F16 1u
#define LM_EXP_DT_BF16 2u
#define LM_EXP_DT_F32 3u
#define LM_EXP_DT_F64 4u
#define LM_EXP_DT_I8 5u
#define LM_EXP_DT_U8 6u
#define LM_EXP_DT_I32 7u
#define LM_EXP_DT_I64 8u
#define LM_EXP_DT_BOOL 9u

#if defined(_WIN32)
#define LM_EXP_EXPORT __declspec(dllexport)
#else
#define LM_EXP_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LmExpBuffer {
  uint32_t struct_size;
  uint32_t dtype;
  void* data;                   /* READ buffers must never be written. */
  uint64_t span_bytes;          /* Accessible storage span including padding. */
  uint32_t rank;
  uint32_t layout;
  uint32_t access;
  uint32_t reserved;            /* Zero. */
  int64_t dims[LM_EXP_MAX_DIMS];
  int64_t strides[LM_EXP_MAX_DIMS]; /* Elements; explicit positive strides. */
} LmExpBuffer;

typedef struct LmExpKernelArgs {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t n_in;
  uint32_t n_out;
  const LmExpBuffer* inputs;
  const LmExpBuffer* outputs;   /* Descriptors immutable; output data writable. */
  const void* params;          /* Opaque bytes; use memcpy, no alignment promise. */
  uint64_t params_bytes;
  void* workspace;             /* Uninitialized scratch, not persistent state. */
  uint64_t workspace_bytes;
} LmExpKernelArgs;

typedef struct LmExpCaps {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t backend;
  uint32_t flags;              /* CPU: SYNCHRONOUS; CUDA: STREAM_ORDERED. */
  const char* kernel;
  const char* implementation;
  const char* entry_symbol;
  const char* numerical_contract;
  uint32_t n_in;
  uint32_t n_out;
  uint64_t params_bytes;
  uint64_t workspace_bytes;
  uint64_t workspace_alignment;
  uint64_t min_buffer_alignment;
} LmExpCaps;

/* CPU: stream must be NULL; return only after all work completes.
 * CUDA: stream is the runtime-owned cudaStream_t (non-NULL). Tensor data and
 * workspace are device pointers. Enqueue ALL work on this stream; do not retain
 * host descriptors/params or destroy the stream. Device buffers live until the
 * runtime synchronizes the stream, including when entry returns an error.
 * Descriptors/params are host pointers, borrowed until entry returns only.
 * Neither backend may throw exceptions across the ABI.
 * 0 = success; 1..255 = kernel error; other returns violate this ABI.
 * data may be NULL only for a zero storage span; do not dereference it then.
 * Caps and their NUL-terminated strings live until the library is unloaded. */
typedef int32_t (*LmExpEntryFn)(const LmExpKernelArgs*, void* stream);
typedef const LmExpCaps* (*LmExpCapsFn)(void);
LM_EXP_EXPORT const LmExpCaps* lm_ku_caps_exp_v1(void);
LM_EXP_EXPORT int32_t lm_ku_entry_exp_v1(const LmExpKernelArgs*, void* stream);

#ifdef __cplusplus
}
#endif
#endif
