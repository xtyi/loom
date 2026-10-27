#ifndef LOOM_EXECUTION_BACKEND_H
#define LOOM_EXECUTION_BACKEND_H
#include "abi/experimental.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace loom::execution {
// 一个 Session 独占一个 Backend。方法返回（包括抛出）前完成所提交的工作，
// 因而公共执行器可安全释放 buffer、params 和动态库；不承诺并发调用。
// 分配结果自己拥有底层资源，释放不能依赖 Backend 对象仍然存活。
class Backend {
 public:
  virtual ~Backend() = default;
  virtual const char* name() const noexcept = 0;
  virtual std::uint32_t abi_backend() const noexcept = 0;
  virtual std::uint32_t abi_flags() const noexcept = 0;
  virtual bool host_accessible() const noexcept = 0;
  virtual std::string device_arch() const { return {}; }
  virtual std::shared_ptr<void> allocate(std::size_t bytes, std::size_t alignment) = 0;
  virtual void upload(void* dst, const void* src, std::size_t bytes) = 0;
  virtual void download(void* dst, const void* src, std::size_t bytes) = 0;
  virtual void zero(void* dst, std::size_t bytes) = 0;
  virtual std::int32_t invoke(LmExpEntryFn entry, const LmExpKernelArgs& args) = 0;
};
std::shared_ptr<void> allocate_host(std::size_t bytes, std::size_t alignment);
} // namespace loom::execution
#endif
