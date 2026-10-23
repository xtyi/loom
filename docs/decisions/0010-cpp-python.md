# 决策 0010 C++17 核心与可选 Python 绑定

- 状态：已接受
- 时间：2026-10-23

## 决定

runtime 与 CLI 使用 C++17，构建采用 CMake，测试采用 CTest。离线 Kernel Harness 使用 Python。面向宿主应用和 KernelUnit 的发布接口采用版本化 C ABI；C++ 类型留在内部，不把 STL 容器、异常或对象布局跨动态库边界传递。

第一版保持 C++17：当前的图、资源管理与加载需求不依赖更新的语言特性，也便于接入已有工具链。后续可根据明确需求升级，不把 C++ 标准版本和 ABI 版本混为一谈。CMake 要求标准版本并关闭编译器扩展，避免悄悄降级。

Python 绑定优先 pybind11，作为可选模块包装公开宿主 API。runtime 本身不链接 Python，也不在内部嵌入解释器。原型首先通过 C++ 测试验证语义，在 session/调用接口具备实际行为后再增加绑定。

## 工程约束

- 内部用 RAII 管理资源；C ABI 使用不透明句柄、明确的所有权、版本与错误返回。C++ 异常不得跨 ABI。
- 长时间执行或等待时，绑定可在不访问 Python 对象的区间释放 GIL；持有 Python buffer 的所有者直到异步执行完成。取消 Python 引用不等于设备已经完成。
- 零拷贝需要 dtype、shape、stride、设备和生命周期都匹配，不能仅传裸指针；设备 tensor 的互操作协议在对应后端阶段验证。
- 可选绑定与第三方依赖不得成为 CPU 基础测试的前提。

## 来源与取舍

[pybind11 入门](https://pybind11.readthedocs.io/en/stable/basics.html) 展示了独立 C++ 实现与 Python 扩展绑定；[GIL 文档](https://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil) 明确不会隐式释放 GIL。

[CMake CXX_STANDARD](https://cmake.org/cmake/help/latest/prop_tgt/CXX_STANDARD.html) 说明标准版本与 REQUIRED/EXTENSIONS 的设置。语言选择服务于现有设备生态和调用边界；它不会自动解决异步所有权或数值正确性问题。
