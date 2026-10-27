#include "execution/session.h"
#include "execution/backend.h"
#include "abi/experimental.h"

#include <algorithm>
#include <dlfcn.h>
#include <limits>
#include <set>
#include <utility>

namespace loom::execution {

Error::Error(ErrorCode category, std::string message,
             std::optional<plan::NodeId> node_id,
             std::optional<plan::SlotId> slot_id, std::int32_t status)
    : std::runtime_error(std::move(message)), code(category), node(node_id),
      slot(slot_id), kernel_status(status) {}

namespace {
using Key = std::pair<std::string, std::string>;
Key key(const plan::EntryRef& ref) { return {ref.kernel, ref.implementation}; }

bool valid_string(const std::string& s) {
  return !s.empty() && s.find('\0') == std::string::npos;
}

struct Library {
  void* handle;
  explicit Library(const std::string& path) : handle(dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL)) {
    if (!handle) throw Error(ErrorCode::load, path + ": " + dlerror());
  }
  ~Library() { dlclose(handle); }
  Library(const Library&) = delete;
  Library& operator=(const Library&) = delete;

  template <class Fn>
  Fn symbol(const std::string& name, const std::string& path) const {
    dlerror();
    void* address = dlsym(handle, name.c_str());
    const char* error = dlerror();
    if (error || !address) {
      throw Error(ErrorCode::symbol, path + ": " + name + ": " +
                  (error ? error : "null symbol"));
    }
    // POSIX dlsym supports conversion to a function pointer.
    return reinterpret_cast<Fn>(address);
  }
};

void check_buffer(BufferView view, const plan::TensorInfo& info, plan::SlotId slot) {
  const auto address = reinterpret_cast<std::uintptr_t>(view.data);
  if (view.bytes < info.span_bytes || (!view.data && info.span_bytes)) {
    throw Error(ErrorCode::capacity, "buffer does not cover tensor storage span", {}, slot);
  }
  if (info.span_bytes && address % info.desc.alignment != 0) {
    throw Error(ErrorCode::alignment, "buffer address violates slot alignment", {}, slot);
  }
  if (info.span_bytes > std::numeric_limits<std::uintptr_t>::max() - address) {
    throw Error(ErrorCode::capacity, "buffer address range overflows", {}, slot);
  }
}

void check_overlap(const std::map<plan::SlotId, BufferView>& views,
                   const plan::PreparedProgram& program) {
  struct Range { std::uintptr_t begin, end; plan::SlotId slot; };
  std::vector<Range> ranges;
  for (const auto& [slot, view] : views) {
    const auto size = program.tensors.at(slot).info.span_bytes;
    if (size) {
      const auto address = reinterpret_cast<std::uintptr_t>(view.data);
      ranges.push_back({address, address + size, slot});
    }
  }
  std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) {
    return a.begin < b.begin;
  });
  for (std::size_t i = 1; i < ranges.size(); ++i) {
    if (ranges[i].begin < ranges[i - 1].end) {
      throw Error(ErrorCode::alias, "buffer overlaps slot " +
                  std::to_string(ranges[i - 1].slot), {}, ranges[i].slot);
    }
  }
}

uint32_t abi_dtype(plan::DType dtype) {
  switch (dtype) {
    case plan::DType::f16: return LM_EXP_DT_F16;
    case plan::DType::bf16: return LM_EXP_DT_BF16;
    case plan::DType::f32: return LM_EXP_DT_F32;
    case plan::DType::f64: return LM_EXP_DT_F64;
    case plan::DType::i8: return LM_EXP_DT_I8;
    case plan::DType::u8: return LM_EXP_DT_U8;
    case plan::DType::i32: return LM_EXP_DT_I32;
    case plan::DType::i64: return LM_EXP_DT_I64;
    case plan::DType::boolean: return LM_EXP_DT_BOOL;
    default: throw Error(ErrorCode::abi, "unsupported dtype");
  }
}

LmExpBuffer buffer(const plan::TensorInfo& info, const void* data, uint32_t access) {
  LmExpBuffer out{};
  out.struct_size = sizeof(out);
  out.dtype = abi_dtype(info.desc.dtype);
  out.data = const_cast<void*>(data);
  out.span_bytes = info.span_bytes;
  out.rank = static_cast<uint32_t>(info.desc.shape.size());
  out.layout = info.desc.layout;
  out.access = access;
  std::copy(info.desc.shape.begin(), info.desc.shape.end(), out.dims);
  std::copy(info.desc.strides.begin(), info.desc.strides.end(), out.strides);
  return out;
}

void check_caps(const LmExpCaps* caps, const plan::PreparedCall& call,
                const std::string& path, const Backend& backend) {
  const auto fail = [&](ErrorCode code, const char* message) {
    throw Error(code, path + ": " + message, call.node);
  };
  if (!caps || caps->struct_size < sizeof(LmExpCaps)) fail(ErrorCode::abi, "missing/truncated caps");
  if (caps->abi_version != LM_EXP_ABI_VERSION) fail(ErrorCode::abi, "experimental ABI version mismatch");
  if (caps->backend != backend.abi_backend() || caps->flags != backend.abi_flags()) {
    fail(ErrorCode::unsupported, "kernel backend or execution flags do not match session");
  }
  const auto& entry = call.entry;
  const auto equal = [](const char* actual, const std::string& wanted) {
    return actual && wanted == actual;
  };
  if (!equal(caps->kernel, entry.ref.kernel) ||
      !equal(caps->implementation, entry.ref.implementation) ||
      !equal(caps->entry_symbol, entry.symbol) ||
      !equal(caps->numerical_contract, entry.numerical_contract)) {
    fail(ErrorCode::contract, "binary identity/symbol/numerical contract mismatch");
  }
  if (caps->n_in != entry.inputs.size() || caps->n_out != entry.outputs.size() ||
      caps->params_bytes != entry.params_bytes ||
      caps->workspace_bytes != entry.workspace_bytes ||
      caps->workspace_alignment != entry.workspace_alignment) {
    fail(ErrorCode::contract, "binary ports/params/workspace contract mismatch");
  }
  if (caps->min_buffer_alignment > std::numeric_limits<std::size_t>::max() ||
      !plan::valid_alignment(static_cast<std::size_t>(caps->min_buffer_alignment))) {
    fail(ErrorCode::abi, "invalid minimum buffer alignment");
  }
  for (const auto* ports : {&entry.inputs, &entry.outputs}) {
    for (const auto& port : *ports) {
      if (port.tensor.alignment < caps->min_buffer_alignment) {
        fail(ErrorCode::contract, "port promises less alignment than binary requires");
      }
    }
  }
}
} // namespace

struct Session::Impl {
  std::unique_ptr<Backend> backend;
  plan::PreparedProgram program;
  std::map<Key, std::unique_ptr<Library>> libraries;
  std::vector<LmExpEntryFn> functions;
  std::map<plan::SlotId, std::shared_ptr<void>> constants;
};

Session::Session(std::unique_ptr<Backend> backend, const plan::Graph& graph,
                 const std::vector<plan::EntryContract>& entries,
                 const std::vector<KernelLibrary>& libraries,
                 const std::map<plan::SlotId, BufferView>& constants,
                 const std::optional<std::vector<plan::NodeId>>& order)
    : impl_(std::make_unique<Impl>()) {
  if (!backend) throw Error(ErrorCode::unsupported, "null backend");
  impl_->backend = std::move(backend);
  auto prepared = plan::prepare_program(graph, entries, order);
  if (!prepared.ok()) {
    Error error(ErrorCode::invalid_program, prepared.error->message,
                prepared.error->node, prepared.error->slot);
    error.diagnostic = std::move(prepared.error);
    throw error;
  }
  impl_->program = std::move(*prepared.value);
  const auto& program = impl_->program;
  const std::set<plan::SlotId> input_ids(program.inputs.begin(), program.inputs.end());
  std::set<plan::SlotId> constant_ids;
  for (const auto& [id, tensor] : program.tensors) {
    if (tensor.external && !input_ids.count(id)) constant_ids.insert(id);
  }
  if (constants.size() != constant_ids.size()) {
    throw Error(ErrorCode::binding, "constants must cover all non-input external slots");
  }
  for (const auto& [id, view] : constants) {
    if (!constant_ids.count(id)) throw Error(ErrorCode::binding, "unexpected constant slot", {}, id);
    check_buffer(view, program.tensors.at(id).info, id);
  }
  check_overlap(constants, program);
  for (const auto& [id, view] : constants) {
    const auto& info = program.tensors.at(id).info;
    auto owned = impl_->backend->allocate(info.span_bytes, info.desc.alignment);
    impl_->backend->upload(owned.get(), view.data, info.span_bytes);
    impl_->constants.emplace(id, std::move(owned));
  }

  std::set<Key> used;
  for (const auto& call : program.calls) used.insert(key(call.entry.ref));
  std::map<Key, std::string> paths;
  for (const auto& library : libraries) {
    if (!used.count(key(library.ref)) || !valid_string(library.path) ||
        !paths.emplace(key(library.ref), library.path).second) {
      throw Error(ErrorCode::invalid_library, "duplicate, unused or invalid library mapping");
    }
  }
  for (const auto& call : program.calls) {
    const auto ref = key(call.entry.ref);
    const auto path_it = paths.find(ref);
    if (path_it == paths.end()) {
      throw Error(ErrorCode::invalid_library, "missing library for " + ref.first + "/" + ref.second, call.node);
    }
    const auto& path = path_it->second;
    auto& library = impl_->libraries[ref];
    try {
      if (!library) library = std::make_unique<Library>(path);
      const auto caps_fn = library->symbol<LmExpCapsFn>("lm_ku_caps_exp_v1", path);
      check_caps(caps_fn(), call, path, *impl_->backend);
      impl_->functions.push_back(library->symbol<LmExpEntryFn>(call.entry.symbol, path));
    } catch (Error& error) {
      error.node = call.node;
      throw;
    }
  }
}

Session::~Session() = default;

RunResult Session::run(const std::vector<BufferView>& inputs) {
  const auto& program = impl_->program;
  if (inputs.size() != program.inputs.size()) {
    throw Error(ErrorCode::binding, "input count differs from graph.inputs");
  }
  std::map<plan::SlotId, BufferView> views;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const auto slot = program.inputs[i];
    check_buffer(inputs[i], program.tensors.at(slot).info, slot);
    views.emplace(slot, inputs[i]);
  }
  check_overlap(views, program);
  std::map<plan::SlotId, std::shared_ptr<void>> owned;
  auto& backend = *impl_->backend;
  // CPU 可直接借用输入；设备后端将输入上传到本次 run 独占的存储。
  if (!backend.host_accessible()) {
    for (auto& [slot, view] : views) {
      const auto& info = program.tensors.at(slot).info;
      auto data = backend.allocate(info.span_bytes, info.desc.alignment);
      backend.upload(data.get(), view.data, info.span_bytes);
      view = {data.get(), info.span_bytes};
      owned.emplace(slot, std::move(data));
    }
  }
  for (const auto& [slot, tensor] : program.tensors) {
    if (!tensor.external) {
      auto data = backend.allocate(tensor.info.span_bytes, tensor.info.desc.alignment);
      // Padding is observable in an owned output's storage span. Initialize it
      // as well; kernels are still required to write every logical output value.
      backend.zero(data.get(), tensor.info.span_bytes);
      views.emplace(slot, BufferView{data.get(), tensor.info.span_bytes});
      owned.emplace(slot, std::move(data));
    } else if (const auto it = impl_->constants.find(slot); it != impl_->constants.end()) {
      views.emplace(slot, BufferView{it->second.get(), tensor.info.span_bytes});
    }
  }
  std::size_t workspace_bytes = 0, workspace_alignment = 1;
  for (const auto& call : program.calls) {
    workspace_bytes = std::max(workspace_bytes, call.entry.workspace_bytes);
    workspace_alignment = std::max(workspace_alignment, call.entry.workspace_alignment);
  }
  const auto workspace = backend.allocate(workspace_bytes, workspace_alignment);
  struct Invocation { std::vector<LmExpBuffer> inputs, outputs; };
  std::vector<Invocation> invocations;
  for (const auto& call : program.calls) {
    Invocation invocation;
    for (const auto id : call.inputs) {
      invocation.inputs.push_back(buffer(program.tensors.at(id).info, views.at(id).data, LM_EXP_READ));
    }
    for (const auto id : call.outputs) {
      invocation.outputs.push_back(buffer(program.tensors.at(id).info, views.at(id).data, LM_EXP_WRITE));
    }
    invocations.push_back(std::move(invocation));
  }
  // 提前分配宿主输出；重复输出共享存储，GPU 中间结果不下载。
  RunResult result;
  std::map<plan::SlotId, std::shared_ptr<void>> outputs;
  for (const auto id : program.outputs) {
    const auto& info = program.tensors.at(id).info;
    if (!outputs.count(id)) {
      outputs.emplace(id, backend.host_accessible() && owned.count(id)
          ? owned.at(id) : allocate_host(info.span_bytes, info.desc.alignment));
    }
    result.outputs.push_back({id, info, outputs.at(id)});
  }
  for (std::size_t i = 0; i < program.calls.size(); ++i) {
    const auto& call = program.calls[i];
    const auto& invocation = invocations[i];
    LmExpKernelArgs args{};
    args.struct_size = sizeof(args);
    args.abi_version = LM_EXP_ABI_VERSION;
    args.n_in = static_cast<uint32_t>(invocation.inputs.size());
    args.n_out = static_cast<uint32_t>(invocation.outputs.size());
    args.inputs = invocation.inputs.empty() ? nullptr : invocation.inputs.data();
    args.outputs = invocation.outputs.empty() ? nullptr : invocation.outputs.data();
    args.params = call.params.empty() ? nullptr : call.params.data();
    args.params_bytes = call.params.size();
    args.workspace = call.entry.workspace_bytes ? workspace.get() : nullptr;
    args.workspace_bytes = call.entry.workspace_bytes;
    std::int32_t status;
    try {
      status = backend.invoke(impl_->functions[i], args);
    } catch (Error& error) {
      error.node = call.node;
      throw;
    }
    if (status != 0) {
      throw Error(status > 0 && status <= 255 ? ErrorCode::kernel : ErrorCode::kernel_protocol,
                  call.entry.ref.kernel + "/" + call.entry.ref.implementation +
                  " returned " + std::to_string(status), call.node, {}, status);
    }
  }
  for (const auto& [id, data] : outputs) {
    if (data.get() != views.at(id).data) {
      backend.download(data.get(), views.at(id).data, program.tensors.at(id).info.span_bytes);
    }
  }
  return result;
}
} // namespace loom::execution
