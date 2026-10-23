cmake_minimum_required(VERSION 3.20)
project(loom VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LOOM_ENABLE_SANITIZERS "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
if(LOOM_ENABLE_SANITIZERS)
  if(MSVC)
    message(FATAL_ERROR "LOOM_ENABLE_SANITIZERS requires GCC or Clang")
  endif()
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer
    -fno-sanitize-recover=all)
  add_link_options(-fsanitize=address,undefined)
endif()

add_library(loom_plan STATIC
  runtime/plan/graph.cpp
  runtime/plan/graph_edit.cpp
  runtime/plan/tensor.cpp
  runtime/plan/program.cpp)
target_include_directories(loom_plan PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/runtime")
if(MSVC)
  target_compile_options(loom_plan PRIVATE /W4)
else()
  target_compile_options(loom_plan PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(loom-plan-demo examples/plan_demo.cpp)
target_link_libraries(loom-plan-demo PRIVATE loom_plan)

add_executable(loom-contract-demo examples/contract_demo.cpp)
target_link_libraries(loom-contract-demo PRIVATE loom_plan)

option(LOOM_ENABLE_EXPERIMENTAL_ABI "Build experimental synchronous CPU loader and kernels" OFF)
if(LOOM_ENABLE_EXPERIMENTAL_ABI)
  if(NOT UNIX)
    message(FATAL_ERROR "Experimental CPU loader currently requires POSIX dlopen/posix_memalign")
  endif()
  enable_language(C)
  set(CMAKE_C_STANDARD 11)
  set(CMAKE_C_STANDARD_REQUIRED ON)
  set(CMAKE_C_EXTENSIONS OFF)
  add_library(loom_cpu STATIC runtime/cpu/session.cpp)
  target_link_libraries(loom_cpu PUBLIC loom_plan PRIVATE ${CMAKE_DL_LIBS})
  target_compile_options(loom_cpu PRIVATE -Wall -Wextra -Wpedantic)

  # Separately compiled KernelUnits: neither links against the runtime.
  function(loom_kernel target source)
    add_library(${target} SHARED ${source})
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/runtime")
    set_target_properties(${target} PROPERTIES C_VISIBILITY_PRESET hidden
      CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
  endfunction()
  loom_kernel(loom-example-add examples/kernels/add.c)
  loom_kernel(loom-example-relu examples/kernels/relu_scale.cpp)
  add_executable(loom-cpu-demo examples/cpu_demo.cpp)
  target_link_libraries(loom-cpu-demo PRIVATE loom_cpu)
  add_dependencies(loom-cpu-demo loom-example-add loom-example-relu)
endif()

include(CTest)
if(BUILD_TESTING)
  if(LOOM_ENABLE_EXPERIMENTAL_ABI)
    add_executable(loom-cpu-test runtime/tests/cpu_test.cpp)
    target_include_directories(loom-cpu-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(loom-cpu-test PRIVATE loom_cpu ${CMAKE_DL_LIBS})
    add_dependencies(loom-cpu-test loom-example-add loom-example-relu)
    add_test(NAME cpu.demo COMMAND loom-cpu-demo
      "$<TARGET_FILE:loom-example-add>" "$<TARGET_FILE:loom-example-relu>")
    foreach(case IN ITEMS execute lifetime output_order shapes bindings constants
        input_order_alias program libraries empty kernel_params)
      add_test(NAME cpu.${case} COMMAND loom-cpu-test "${case}"
        "$<TARGET_FILE:loom-example-add>" "$<TARGET_FILE:loom-example-relu>")
    endforeach()
    set(fixture_mode 0)
    foreach(case IN ITEMS caps_version caps_truncated caps_backend caps_async
        caps_identity caps_numerics caps_alignment caps_ports caps_workspace
        caps_params missing_entry missing_caps caps_null kernel_protocol fail_fast
        caps_flags caps_stronger_alignment)
      math(EXPR fixture_mode "${fixture_mode} + 1")
      loom_kernel(loom-fixture-${case} runtime/tests/cpu_kernel_fixture.c)
      target_compile_definitions(loom-fixture-${case} PRIVATE FIXTURE_MODE=${fixture_mode})
      add_dependencies(loom-cpu-test loom-fixture-${case})
      add_test(NAME cpu.${case} COMMAND loom-cpu-test "${case}"
        "$<TARGET_FILE:loom-example-add>" "$<TARGET_FILE:loom-example-relu>"
        "$<TARGET_FILE:loom-fixture-${case}>")
    endforeach()
  endif()

  add_executable(loom-graph-edit-test runtime/tests/graph_edit_test.cpp)
  target_link_libraries(loom-graph-edit-test PRIVATE loom_plan)
  foreach(case IN ITEMS create modify invalid_edits editor_lifetime remove
      global_validation program)
    add_test(NAME graph_edit.${case} COMMAND loom-graph-edit-test "${case}")
  endforeach()

  add_executable(loom-plan-test runtime/tests/plan_test.cpp)
  target_link_libraries(loom-plan-test PRIVATE loom_plan)
  foreach(case IN ITEMS
      fork_join deterministic explicit_order invalid_order duplicate_schedule
      missing_schedule unknown_schedule duplicate_node duplicate_slot
      unknown_input unknown_output unknown_after unknown_graph_output graph_inputs
      missing_producer multiple_producers external_write self_dependency
      data_cycle after_cycle duplicate_edges empty_graph external_output
      zero_and_large_ids)
    add_test(NAME plan.${case} COMMAND loom-plan-test "${case}")
  endforeach()

  add_executable(loom-tensor-test runtime/tests/tensor_test.cpp)
  target_link_libraries(loom-tensor-test PRIVATE loom_plan)
  foreach(case IN ITEMS storage_types compact scalar zero_extent padded transpose
      singleton invalid_shape invalid_strides alignment overflow)
    add_test(NAME tensor.${case} COMMAND loom-tensor-test "${case}")
  endforeach()

  add_executable(loom-program-test runtime/tests/program_test.cpp)
  target_link_libraries(loom-program-test PRIVATE loom_plan)
  foreach(case IN ITEMS mixed_plan graph_inputs missing_metadata registry invalid_ports arity
      port_order dtype shape layout strides alignment numerics params snapshot
      topology empty)
    add_test(NAME program.${case} COMMAND loom-program-test "${case}")
  endforeach()
endif()
