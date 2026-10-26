#ifndef LOOM_IO_PACKAGE_H
#define LOOM_IO_PACKAGE_H

#include "io/json.h"

#include <map>
#include <memory>

namespace loom::io {

struct Target {
  std::string os;
  std::string arch;
  std::string endianness;
  std::size_t pointer_bits;
};
Target host_target();

struct Library {
  plan::EntryRef ref;
  std::filesystem::path path;
};

struct TensorData {
  std::shared_ptr<const void> data;
  std::size_t bytes;
};

struct Package {
  std::string name;
  std::filesystem::path root;
  PlanDocument plan;
  std::vector<plan::EntryContract> entries;
  std::vector<Library> libraries;
  std::map<plan::SlotId, TensorData> constants;
  plan::PreparedProgram program; // Validated snapshot for inspection, never an execution credential.
};

// Directory package with package.json; checks target, all documents, bindings,
// contained regular-file paths and exact constant storage spans. Never dlopen.
Package load_package(const std::filesystem::path& directory);

// Raw native tensor storage, including stride padding; returns aligned ownership.
TensorData load_tensor(const std::filesystem::path& path, const plan::TensorInfo& info);

} // namespace loom::io
#endif
