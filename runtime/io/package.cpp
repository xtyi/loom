#include "io/package.h"
#include "io/json_internal.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <new>
#include <set>

namespace loom::io {
namespace fs = std::filesystem;
namespace {
using detail::Json;
using detail::Reader;
using Key = std::pair<std::string, std::string>;
Key key(const plan::EntryRef& ref) { return {ref.kernel, ref.implementation}; }

fs::path member(const fs::path& root, const std::string& name,
                const Reader& r, const std::string& field) {
  const fs::path relative(name);
  // Paths in every document are relative to the package root, not the manifest.
  if (relative.empty() || relative.is_absolute() || relative.has_root_name() ||
      name.find('\\') != std::string::npos || name.find(':') != std::string::npos) {
    r.fail(ErrorCode::path, field, "expected package-relative path using /");
  }
  for (const auto& part : relative) {
    if (part == "..") r.fail(ErrorCode::path, field, "parent path components are forbidden");
  }
  std::error_code ec;
  const auto path = fs::canonical(root / relative, ec);
  if (ec) r.fail(ErrorCode::io, field, "cannot resolve " + name + ": " + ec.message());
  auto base = root.begin();
  auto child = path.begin();
  for (; base != root.end() && child != path.end() && *base == *child; ++base, ++child) {}
  if (base != root.end() || child == path.end()) {
    r.fail(ErrorCode::path, field, "path/symlink escapes package directory: " + name);
  }
  if (!fs::is_regular_file(path, ec) || ec) {
    r.fail(ErrorCode::path, field, "expected regular file: " + name);
  }
  return path;
}

void target(const Json& j, const Reader& r) {
  r.object(j, "$.target", {"backend", "abi", "os", "arch", "endianness", "pointer_bits"});
  const auto host = host_target();
  if (host.os == "unknown" || host.arch == "unknown") {
    r.fail(ErrorCode::target, "$.target", "this host target is not supported by the experimental format");
  }
  const auto equal = [&](const char* field, const std::string& wanted) {
    const auto path = std::string("$.target.") + field;
    if (r.string(j.at(field), path) != wanted) r.fail(ErrorCode::target, path, "requires " + wanted);
  };
  equal("backend", "cpu");
  equal("abi", "loom.cpu.exp.v1");
  equal("os", host.os);
  equal("arch", host.arch);
  equal("endianness", host.endianness);
  if (r.uint(j.at("pointer_bits"), "$.target.pointer_bits", 128) != host.pointer_bits) {
    r.fail(ErrorCode::target, "$.target.pointer_bits", "native pointer width mismatch");
  }
}
} // namespace

Target host_target() {
  Target t;
#if defined(__linux__)
  t.os = "linux";
#elif defined(__APPLE__)
  t.os = "macos";
#elif defined(_WIN32)
  t.os = "windows";
#else
  t.os = "unknown";
#endif
#if defined(__x86_64__) || defined(_M_X64)
  t.arch = "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
  t.arch = "aarch64";
#elif defined(__i386__) || defined(_M_IX86)
  t.arch = "x86";
#elif defined(__arm__) || defined(_M_ARM)
  t.arch = "arm";
#else
  t.arch = "unknown";
#endif
  const std::uint16_t one = 1;
  t.endianness = *reinterpret_cast<const unsigned char*>(&one) ? "little" : "big";
  t.pointer_bits = sizeof(void*) * 8;
  return t;
}

TensorData load_tensor(const fs::path& path, const plan::TensorInfo& info) {
  const Reader r{path.string()};
  std::error_code ec;
  const auto bytes = fs::file_size(path, ec);
  if (ec) r.fail(ErrorCode::io, "$", "cannot stat tensor file: " + ec.message());
  if (bytes != info.span_bytes) r.fail(ErrorCode::binding, "$", "tensor file size must equal storage span " + std::to_string(info.span_bytes));
  if (bytes > static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max())) {
    r.fail(ErrorCode::binding, "$", "tensor file exceeds supported stream size");
  }
  const auto alignment = std::max(info.desc.alignment, alignof(std::max_align_t));
  void* raw = bytes ? ::operator new(static_cast<std::size_t>(bytes), std::align_val_t(alignment)) : nullptr;
  std::shared_ptr<void> data(raw, [alignment](void* p) { ::operator delete(p, std::align_val_t(alignment)); });
  std::ifstream in(path, std::ios::binary);
  if (!in || (bytes && !in.read(static_cast<char*>(raw), static_cast<std::streamsize>(bytes))) ||
      in.peek() != std::char_traits<char>::eof()) {
    r.fail(ErrorCode::io, "$", "cannot read complete tensor file");
  }
  return {std::move(data), static_cast<std::size_t>(bytes)};
}

Package load_package(const fs::path& directory) {
  std::error_code ec;
  const auto root = fs::canonical(directory, ec);
  if (ec || !fs::is_directory(root)) throw Error(ErrorCode::io, directory.string(), "$", "expected package directory");
  const auto manifest = member(root, "package.json", Reader{directory.string()}, "$");
  const Reader r{manifest.string()};
  const auto j = detail::parse(detail::read_text(manifest), r);
  r.header(j, package_format, {"static-tensors", "single-stream", "cpu-sync"});
  r.object(j, "$", {"format", "name", "target", "plan", "kernels", "constants"}, {"required_capabilities", "meta"});
  target(j.at("target"), r); // Reject incompatible native data/ABI before loading other files.
  Package out;
  out.root = root;
  out.name = r.string(j.at("name"), "$.name");
  const auto plan_path = member(root, r.string(j.at("plan"), "$.plan"), r, "$.plan");
  out.plan = load_plan(plan_path);
  const auto& kernels = r.array(j.at("kernels"), "$.kernels");
  for (std::size_t i = 0; i < kernels.size(); ++i) {
    const auto field = "$.kernels[" + std::to_string(i) + "]";
    const auto path = member(root, r.string(kernels[i], field), r, field);
    const Reader kr{path.string()};
    const auto kernel = detail::parse(detail::read_text(path), kr);
    auto contract = detail::entry(kernel, kr);
    out.libraries.push_back({contract.ref, member(root, kr.string(kernel.at("library"), "$.library"), kr, "$.library")});
    out.entries.push_back(std::move(contract));
  }
  auto result = plan::prepare_program(out.plan.graph, out.entries, out.plan.order);
  if (!result.ok()) detail::program_error(Reader{plan_path.string()}, *result.error);
  out.program = std::move(*result.value);
  std::set<Key> used;
  for (const auto& call : out.program.calls) used.insert(key(call.entry.ref));
  for (const auto& library : out.libraries) {
    if (!used.count(key(library.ref))) r.fail(ErrorCode::binding, "$.kernels", "unused kernel manifest");
  }
  const std::set<plan::SlotId> inputs(out.program.inputs.begin(), out.program.inputs.end());
  std::set<plan::SlotId> wanted;
  for (const auto& [id, tensor] : out.program.tensors) {
    if (tensor.external && !inputs.count(id)) wanted.insert(id);
  }
  const auto& constants = r.array(j.at("constants"), "$.constants");
  // First validate the entire binding table, then allocate/read any weight data.
  std::map<plan::SlotId, fs::path> files;
  for (std::size_t i = 0; i < constants.size(); ++i) {
    const auto field = "$.constants[" + std::to_string(i) + "]";
    const auto& constant = constants[i];
    r.object(constant, field, {"slot", "file"});
    const auto id = static_cast<plan::SlotId>(r.uint(constant.at("slot"), field + ".slot", UINT32_MAX));
    if (!wanted.count(id) || files.count(id)) r.fail(ErrorCode::binding, field + ".slot", "unexpected or duplicate constant slot");
    files.emplace(id, member(root, r.string(constant.at("file"), field + ".file"), r, field + ".file"));
  }
  if (files.size() != wanted.size()) r.fail(ErrorCode::binding, "$.constants", "must cover every non-input external slot exactly once");
  for (const auto& [id, file] : files) {
    out.constants.emplace(id, load_tensor(file, out.program.tensors.at(id).info));
  }
  return out;
}
} // namespace loom::io
