#include "cpu/backend.h"
#include <cstring>
namespace loom::cpu {
namespace {
class Backend final : public execution::Backend {
 public:
  const char* name() const noexcept override { return "cpu"; }
  std::uint32_t abi_backend() const noexcept override { return LM_EXP_BACKEND_CPU; }
  std::uint32_t abi_flags() const noexcept override { return LM_EXP_SYNCHRONOUS; }
  bool host_accessible() const noexcept override { return true; }
  std::shared_ptr<void> allocate(std::size_t bytes, std::size_t alignment) override {
    return execution::allocate_host(bytes, alignment);
  }
  void upload(void* dst, const void* src, std::size_t bytes) override {
    if (bytes) std::memcpy(dst, src, bytes);
  }
  void download(void* dst, const void* src, std::size_t bytes) override { upload(dst, src, bytes); }
  void zero(void* dst, std::size_t bytes) override { if (bytes) std::memset(dst, 0, bytes); }
  std::int32_t invoke(LmExpEntryFn entry, const LmExpKernelArgs& args) override {
    return entry(&args, nullptr);
  }
};
}
std::unique_ptr<execution::Backend> make_backend() { return std::make_unique<Backend>(); }
} // namespace loom::cpu
