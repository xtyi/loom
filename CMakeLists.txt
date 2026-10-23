cmake_minimum_required(VERSION 3.20)
project(loom VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

include(CTest)
if(BUILD_TESTING)
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
