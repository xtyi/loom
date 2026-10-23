#include "plan/tensor.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace loom::plan {
namespace {

TensorResult fail(TensorErrorCode code, std::string field, std::string message) {
  return {std::nullopt, TensorError{code, std::move(field), std::move(message)}};
}

std::size_t element_bytes(DType dtype) {
  switch (dtype) {
    case DType::boolean:
    case DType::i8:
    case DType::u8: return 1;
    case DType::f16:
    case DType::bf16: return 2;
    case DType::f32:
    case DType::i32: return 4;
    case DType::f64:
    case DType::i64: return 8;
    default: return 0;
  }
}

// Keep dimensions/strides/extents representable in both size_t and the signed
// 64-bit descriptor domain, even on hosts with a smaller address space.
constexpr std::uint64_t max_extent =
    std::numeric_limits<std::size_t>::max() <
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())
        ? std::numeric_limits<std::size_t>::max()
        : static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());

bool multiply(std::uint64_t a, std::uint64_t b, std::uint64_t& result) {
  if (b != 0 && a > max_extent / b) {
    return false;
  }
  result = a * b;
  return true;
}

}  // namespace

const char* dtype_name(DType dtype) {
  switch (dtype) {
    case DType::f16: return "f16";
    case DType::bf16: return "bf16";
    case DType::f32: return "f32";
    case DType::f64: return "f64";
    case DType::i8: return "i8";
    case DType::u8: return "u8";
    case DType::i32: return "i32";
    case DType::i64: return "i64";
    case DType::boolean: return "bool";
    default: return "invalid/unsupported";
  }
}

bool valid_alignment(std::size_t alignment) {
  return alignment != 0 && (alignment & (alignment - 1)) == 0;
}

TensorResult validate_tensor(const TensorDesc& desc) {
  const auto bytes = element_bytes(desc.dtype);
  if (bytes == 0) {
    return fail(TensorErrorCode::unsupported_dtype, "dtype",
                "unknown or unsupported storage dtype");
  }
  if (desc.shape.size() > 8) {
    return fail(TensorErrorCode::invalid_shape, "shape", "rank exceeds 8");
  }
  if (!valid_alignment(desc.alignment)) {
    return fail(TensorErrorCode::invalid_alignment, "alignment",
                "alignment must be a nonzero power of two");
  }
  bool empty = false;
  for (const auto dimension : desc.shape) {
    if (dimension < 0 || static_cast<std::uint64_t>(dimension) > max_extent) {
      return fail(TensorErrorCode::invalid_shape, "shape",
                  "dimensions must be nonnegative static host-sized integers");
    }
    empty = empty || dimension == 0;
  }

  TensorDesc normalized = desc;
  if (desc.strides.empty()) {
    normalized.strides.resize(desc.shape.size());
    std::uint64_t stride = 1;
    for (std::size_t i = desc.shape.size(); i > 0; --i) {
      normalized.strides[i - 1] = static_cast<std::int64_t>(stride);
      // No stride for a preceding dimension is needed at i == 1.
      if (i > 1 && !multiply(stride,
                            static_cast<std::uint64_t>(
                                std::max<std::int64_t>(desc.shape[i - 1], 1)),
                            stride)) {
        return fail(TensorErrorCode::size_overflow, "strides",
                    "compact strides exceed the supported extent");
      }
    }
  } else if (desc.strides.size() != desc.shape.size()) {
    return fail(TensorErrorCode::invalid_strides, "strides",
                "stride count must equal rank");
  }
  for (const auto stride : normalized.strides) {
    if (stride <= 0 || static_cast<std::uint64_t>(stride) > max_extent) {
      return fail(TensorErrorCode::invalid_strides, "strides",
                  "only positive host-sized strides are supported");
    }
  }

  if (empty) {
    return {TensorInfo{std::move(normalized), bytes, 0, 0}, std::nullopt};
  }
  std::uint64_t count = 1;
  // Ignore singleton axes for overlap checks; their stride is never traversed.
  std::vector<std::pair<std::uint64_t, std::uint64_t>> axes;
  for (std::size_t i = 0; i < desc.shape.size(); ++i) {
    const auto dimension = static_cast<std::uint64_t>(desc.shape[i]);
    if (!multiply(count, dimension, count)) {
      return fail(TensorErrorCode::size_overflow, "shape",
                  "element count exceeds the supported extent");
    }
    if (dimension > 1) {
      axes.emplace_back(normalized.strides[i], dimension);
    }
  }
  std::sort(axes.begin(), axes.end());
  std::uint64_t span = 1;
  for (const auto& [stride, dimension] : axes) {
    // Sufficient, conservative proof of non-overlap; exotic interleavings that
    // fail this test are unsupported even if an exhaustive proof could work.
    if (stride < span) {
      return fail(TensorErrorCode::invalid_strides, "strides",
                  "strides do not prove nonoverlapping storage");
    }
    std::uint64_t extension;
    if (!multiply(dimension - 1, stride, extension) ||
        extension > max_extent - span) {
      return fail(TensorErrorCode::size_overflow, "strides",
                  "strided span exceeds the supported extent");
    }
    span += extension;
  }
  if (!multiply(span, bytes, span)) {
    return fail(TensorErrorCode::size_overflow, "dtype",
                "byte span exceeds the supported extent");
  }
  return {TensorInfo{std::move(normalized), bytes, static_cast<std::size_t>(count),
                     static_cast<std::size_t>(span)}, std::nullopt};
}

}  // namespace loom::plan
