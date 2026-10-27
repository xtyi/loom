# 可重定位 CUDA 包复用同一图、数据和 manifest 模板。
# 示例包只标记一个精确 SM；交叉编译时可显式指定，不依赖运行时探测。
set(LOOM_CUDA_PACKAGE_SM "" CACHE STRING "Exact SM of the example CUDA package (e.g. 120)")
if(NOT LOOM_CUDA_PACKAGE_SM)
  list(GET CMAKE_CUDA_ARCHITECTURES 0 LOOM_CUDA_PACKAGE_SM)
  string(REGEX REPLACE "-(real|virtual)$" "" LOOM_CUDA_PACKAGE_SM "${LOOM_CUDA_PACKAGE_SM}")
endif()
if(NOT LOOM_CUDA_PACKAGE_SM MATCHES "^[0-9]+$")
  message(FATAL_ERROR "Set LOOM_CUDA_PACKAGE_SM to a numeric SM when using native/all architectures")
endif()
set(LOOM_CUDA_EXAMPLE_PACKAGE "${CMAKE_CURRENT_BINARY_DIR}/examples/cuda.loom")
loom_example_package(cuda loom-example-cuda "${LOOM_CUDA_EXAMPLE_PACKAGE}"
  loom-example-cuda-package "${LOOM_CUDA_PACKAGE_SM}")
