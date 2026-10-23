#ifndef LOOM_PLAN_TENSOR_H
#define LOOM_PLAN_TENSOR_H

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace loom::plan {

// S3 metadata types, not the published ABI. Recognizing a storage dtype does
// not imply that a backend has an implementation for it.
enum class DType { invalid, f16, bf16, f32, f64, i8, u8, i32, i64, boolean };

struct TensorDesc {
  DType dtype = DType::invalid;
  std::vector<std::int64_t> shape;    // Rank 0 is a scalar; zero extents allowed.
  std::vector<std::int64_t> strides;  // Elements; empty means compact row-major.
  std::uint32_t layout = 0;           // Opaque label, including 0; exact match.
  std::size_t alignment = 1;         // Promised base alignment, in bytes.
};

struct TensorInfo {
  TensorDesc desc;  // Owns a normalized descriptor with explicit strides.
  std::size_t element_bytes;
  std::size_t elements;
  std::size_t span_bytes;  // Includes holes/padding, but no trailing alignment.
};

enum class TensorErrorCode {
  unsupported_dtype,
  invalid_shape,
  invalid_strides,
  invalid_alignment,
  size_overflow,
};

struct TensorError {
  TensorErrorCode code;
  std::string field;
  std::string message;
};

struct TensorResult {
  std::optional<TensorInfo> value;
  std::optional<TensorError> error;
  bool ok() const { return value.has_value(); }
};

const char* dtype_name(DType dtype);
bool valid_alignment(std::size_t alignment);

// Static, byte-addressable, rank <= 8. Supports positive, provably nonoverlapping
// strided storage (including permutations/padding). No broadcasts, negative
// strides, sub-byte packing, offsets or aliases. Never touches actual memory.
TensorResult validate_tensor(const TensorDesc& desc);

}  // namespace loom::plan

#endif
