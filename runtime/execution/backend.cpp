#include "execution/backend.h"
#include <algorithm>
#include <new>

namespace loom::execution {
std::shared_ptr<void> allocate_host(std::size_t bytes, std::size_t alignment) {
  if (!bytes) return {};
  alignment = std::max(alignment, alignof(std::max_align_t));
  return {::operator new(bytes, std::align_val_t(alignment)),
          [alignment](void* p) { ::operator delete(p, std::align_val_t(alignment)); }};
}
} // namespace loom::execution
