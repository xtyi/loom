#ifndef LOOM_IO_JSON_INTERNAL_H
#define LOOM_IO_JSON_INTERNAL_H

#include <nlohmann/json.hpp>

#include "io/json.h"

namespace loom::io::detail {
using Json = nlohmann::json;

// Typed schema decoding uses a third-party JSON parser; never coerce numbers.
struct Reader {
  std::string source;
  [[noreturn]] void fail(ErrorCode code, const std::string& field,
                         const std::string& message) const;
  // Require an object with all required keys and no keys outside these lists.
  // Field values are validated separately by the typed readers below.
  void validate_object_fields(const Json& j, const std::string& field,
                              std::initializer_list<const char*> required,
                              std::initializer_list<const char*> optional = {}) const;
  const Json& array(const Json& j, const std::string& field) const;
  std::string string(const Json& j, const std::string& field) const;
  std::uint64_t uint(const Json& j, const std::string& field, std::uint64_t max) const;
  bool boolean(const Json& j, const std::string& field) const;
  // Require an exact format and supported required_capabilities; optional meta must be an object.
  void validate_format_and_capabilities(const Json& j, const char* format,
                                        std::initializer_list<const char*> capabilities) const;
  plan::TensorDesc tensor(const Json& j, const std::string& field) const;
  plan::EntryRef ref(const Json& j, const std::string& field) const;
  std::vector<std::uint32_t> ids(const Json& j, const std::string& field) const;
};

Json parse(std::string_view text, const Reader& reader);
std::string read_text(const std::filesystem::path& path);
void program_error(const Reader& reader, const plan::Diagnostic& diagnostic);
plan::EntryContract entry(const Json& j, const Reader& reader);

}  // namespace loom::io::detail
#endif
