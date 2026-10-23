#include "plan/tensor.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>

using namespace loom::plan;

namespace {

void require(bool value, const std::string& message) {
  if (!value) throw std::runtime_error(message);
}

TensorInfo good(const TensorDesc& desc) {
  const auto result = validate_tensor(desc);
  require(result.ok(), result.error ? result.error->message : "missing result");
  require(!result.error, "successful result has an error");
  return *result.value;
}

void bad(const TensorDesc& desc, TensorErrorCode code) {
  const auto result = validate_tensor(desc);
  require(!result.ok() && result.error.has_value(), "invalid tensor accepted");
  require(result.error->code == code, "wrong error code");
  require(!result.error->field.empty() && !result.error->message.empty(),
          "incomplete diagnostic");
}

const std::map<std::string, std::function<void()>> cases{
    {"storage_types", [] {
       const std::map<DType, std::size_t> widths{
           {DType::f16, 2}, {DType::bf16, 2}, {DType::f32, 4}, {DType::f64, 8},
           {DType::i8, 1}, {DType::u8, 1}, {DType::i32, 4}, {DType::i64, 8},
           {DType::boolean, 1}};
       for (const auto& [dtype, bytes] : widths) {
         const auto info = good({dtype, {3}});
         require(info.element_bytes == bytes && info.span_bytes == 3 * bytes,
                 "incorrect storage width");
       }
       bad({}, TensorErrorCode::unsupported_dtype);
       bad({static_cast<DType>(999), {2}}, TensorErrorCode::unsupported_dtype);
     }},
    {"compact", [] {
       const auto info = good({DType::f32, {2, 3, 4}});
       require(info.elements == 24 && info.span_bytes == 96, "incorrect size");
       require(info.desc.strides == std::vector<std::int64_t>({12, 4, 1}),
               "incorrect compact strides");
     }},
    {"scalar", [] {
       const auto info = good({DType::f16, {}});
       require(info.elements == 1 && info.span_bytes == 2 && info.desc.strides.empty(),
               "scalar must contain one element");
       bad({DType::f32, {}, {1}}, TensorErrorCode::invalid_strides);
     }},
    {"zero_extent", [] {
       for (const auto shape : {std::vector<std::int64_t>{0, 3},
                                std::vector<std::int64_t>{3, 0}}) {
         const auto info = good({DType::f32, shape});
         require(info.elements == 0 && info.span_bytes == 0, "nonempty zero tensor");
       }
       const auto info = good({DType::f32, {2, 0, 3}});
       require(info.desc.strides == std::vector<std::int64_t>({3, 3, 1}),
               "zero extents must use max(dim, 1) for compact strides");
       bad({DType::f32, {0, 3}, {0, 1}}, TensorErrorCode::invalid_strides);
     }},
    {"padded", [] {
       const auto info = good({DType::f32, {2, 3}, {4, 1}, 7, 16});
       require(info.elements == 6 && info.span_bytes == 28,
               "span must include interior padding, not trailing padding");
       require(info.desc.layout == 7 && info.desc.alignment == 16, "lost metadata");
     }},
    {"transpose", [] {
       const auto info = good({DType::f32, {2, 3}, {1, 2}});
       require(info.elements == 6 && info.span_bytes == 24, "invalid transpose span");
       require(info.desc.strides == std::vector<std::int64_t>({1, 2}),
               "explicit strides were rewritten");
     }},
    {"singleton", [] {
       const auto info = good({DType::f32, {2, 1, 3}, {3, 999, 1}});
       require(info.span_bytes == 24, "singleton stride inflated storage span");
     }},
    {"invalid_shape", [] {
       bad({DType::f32, {-1, 3}}, TensorErrorCode::invalid_shape);
       bad({DType::f32, {0, -1}}, TensorErrorCode::invalid_shape);
       bad({DType::f32, std::vector<std::int64_t>(9, 1)},
           TensorErrorCode::invalid_shape);
       require(good({DType::u8, std::vector<std::int64_t>(8, 1)}).elements == 1,
               "rank 8 rejected");
     }},
    {"invalid_strides", [] {
       bad({DType::f32, {2, 3}, {1}}, TensorErrorCode::invalid_strides);
       bad({DType::f32, {2, 3}, {0, 1}}, TensorErrorCode::invalid_strides);
       bad({DType::f32, {2, 3}, {-3, 1}}, TensorErrorCode::invalid_strides);
       bad({DType::f32, {2, 3}, {2, 1}}, TensorErrorCode::invalid_strides);
       // Nonoverlapping but exotic interleaving: intentionally conservative.
       bad({DType::u8, {3, 2}, {2, 3}}, TensorErrorCode::invalid_strides);
     }},
    {"alignment", [] {
       bad({DType::f32, {1}, {}, 0, 0}, TensorErrorCode::invalid_alignment);
       bad({DType::f32, {1}, {}, 0, 3}, TensorErrorCode::invalid_alignment);
       require(good({DType::u8, {1}, {}, 0, 64}).desc.alignment == 64,
               "valid alignment rejected");
     }},
    {"overflow", [] {
       const auto max = static_cast<std::int64_t>(std::min<std::uint64_t>(
           std::numeric_limits<std::size_t>::max(),
           std::numeric_limits<std::int64_t>::max()));
       bad({DType::u8, {max, 2}}, TensorErrorCode::size_overflow);
       bad({DType::f64, {max / 8 + 1}}, TensorErrorCode::size_overflow);
       bad({DType::u8, {1, max, 2}}, TensorErrorCode::size_overflow);
       bad({DType::u8, {2}, {max}}, TensorErrorCode::size_overflow);
       bad({DType::u8, {3}, {max}}, TensorErrorCode::size_overflow);
       bad({DType::u8, {0, max, 2}}, TensorErrorCode::size_overflow);
     }},
};

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 || cases.count(argv[1]) == 0) return 2;
  try {
    cases.at(argv[1])();
  } catch (const std::exception& error) {
    std::cerr << argv[1] << ": " << error.what() << '\n';
    return 1;
  }
}
