#include "nv/backend.h"
#include "execution/session.h"
#include <cuda_runtime_api.h>
#include <limits>
#include <string>

namespace loom::nv {
namespace {
void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw execution::Error(execution::ErrorCode::device,
        std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

class DeviceScope {
 public:
  explicit DeviceScope(int device) {
    check(cudaGetDevice(&previous_), "cudaGetDevice");
    if (previous_ != device) check(cudaSetDevice(device), "cudaSetDevice");
  }
  ~DeviceScope() { (void)cudaSetDevice(previous_); }
 private:
  int previous_ = 0;
};

class Backend final : public execution::Backend {
 public:
  explicit Backend(int device) : device_(device) {
    DeviceScope scope(device_);
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, device_), "cudaGetDeviceProperties");
    sm_ = prop.major * 10 + prop.minor;
    check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
  }
  ~Backend() override {
    try {
      DeviceScope scope(device_);
      (void)cudaStreamSynchronize(stream_);
      (void)cudaStreamDestroy(stream_);
    } catch (...) {} // 析构不能掩盖原始执行错误。
  }
  const char* name() const noexcept override { return "cuda"; }
  std::uint32_t abi_backend() const noexcept override { return LM_EXP_BACKEND_CUDA; }
  std::uint32_t abi_flags() const noexcept override { return LM_EXP_STREAM_ORDERED; }
  bool host_accessible() const noexcept override { return false; }
  std::string device_arch() const override { return "sm_" + std::to_string(sm_); }

  std::shared_ptr<void> allocate(std::size_t bytes, std::size_t alignment) override {
    if (!bytes) return {};
    if (!alignment || (alignment & (alignment - 1)) ||
        bytes > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
      throw execution::Error(execution::ErrorCode::capacity, "invalid CUDA allocation size/alignment");
    }
    DeviceScope scope(device_);
    void* base = nullptr;
    check(cudaMalloc(&base, bytes + alignment - 1), "cudaMalloc");
    // 支持大于 cudaMalloc 默认保证的 slot/workspace 对齐，释放原始地址。
    std::shared_ptr<void> owner(base, [device = device_](void* p) {
      try { DeviceScope scope(device); (void)cudaFree(p); } catch (...) {}
    });
    const auto address = reinterpret_cast<std::uintptr_t>(base);
    const auto aligned = (address + alignment - 1) & ~(std::uintptr_t(alignment) - 1);
    return {owner, reinterpret_cast<void*>(aligned)};
  }
  void upload(void* dst, const void* src, std::size_t bytes) override {
    copy(dst, src, bytes, cudaMemcpyHostToDevice);
  }
  void download(void* dst, const void* src, std::size_t bytes) override {
    copy(dst, src, bytes, cudaMemcpyDeviceToHost);
  }
  void zero(void* dst, std::size_t bytes) override {
    if (!bytes) return;
    DeviceScope scope(device_);
    const auto status = cudaMemsetAsync(dst, 0, bytes, stream_);
    const auto completed = cudaStreamSynchronize(stream_);
    check(status, "cudaMemsetAsync");
    check(completed, "cudaStreamSynchronize(zero)");
  }
  std::int32_t invoke(LmExpEntryFn entry, const LmExpKernelArgs& args) override {
    DeviceScope scope(device_);
    const auto status = entry(&args, stream_);
    // 即使 launcher 返回错误，也先等已提交的工作结束，再允许释放资源。
    check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize(kernel)");
    return status;
  }
 private:
  void copy(void* dst, const void* src, std::size_t bytes, cudaMemcpyKind kind) {
    if (!bytes) return;
    DeviceScope scope(device_);
    const auto status = cudaMemcpyAsync(dst, src, bytes, kind, stream_);
    const auto completed = cudaStreamSynchronize(stream_);
    check(status, "cudaMemcpyAsync");
    check(completed, "cudaStreamSynchronize(copy)");
  }
  int device_, sm_ = 0;
  cudaStream_t stream_ = nullptr;
};
}
std::unique_ptr<execution::Backend> make_backend(int device) { return std::make_unique<Backend>(device); }
} // namespace loom::nv
