#include "plan/graph.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>

using namespace loom::plan;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

Graph diamond() {
  return {{{0, true}, {1}, {2}, {3}, {4}},
          {{5, {2, 3}, {4}, {}},
           {20, {1}, {3}, {}},
           {40, {0}, {1}, {}},
           {10, {1}, {2}, {}}},
          {0}, {4}};
}

void expect_order(const PrepareResult& result, std::vector<NodeId> expected) {
  require(result.ok(), result.error ? result.error->message : "prepare failed");
  require(result.order == expected, "unexpected submission order");
}

void expect_error(const PrepareResult& result, ErrorCode expected,
                  std::optional<NodeId> node = std::nullopt,
                  std::optional<SlotId> slot = std::nullopt) {
  require(!result.ok(), "invalid graph/schedule accepted");
  require(result.error->code == expected, "unexpected error category");
  require(!result.error->message.empty(), "missing diagnostic message");
  require(result.order.empty(), "failed validation returned a partial schedule");
  if (node) {
    require(result.error->node == node, "wrong diagnostic node");
  }
  if (slot) {
    require(result.error->slot == slot, "wrong diagnostic slot");
  }
}

const std::map<std::string, std::function<void()>> cases{
    {"fork_join", [] {
       expect_order(prepare(diamond()), {40, 10, 20, 5});
     }},
    {"deterministic", [] {
       auto graph = diamond();
       const auto less = [](const Node& a, const Node& b) { return a.id < b.id; };
       std::sort(graph.nodes.begin(), graph.nodes.end(), less);
       do {
         std::reverse(graph.slots.begin(), graph.slots.end());
         expect_order(prepare(graph), {40, 10, 20, 5});
       } while (std::next_permutation(graph.nodes.begin(), graph.nodes.end(), less));
     }},
    {"explicit_order", [] {
       expect_order(prepare(diamond(), std::vector<NodeId>{40, 20, 10, 5}),
                    {40, 20, 10, 5});
     }},
    {"invalid_order", [] {
       expect_error(prepare(diamond(), std::vector<NodeId>{5, 10, 20, 40}),
                    ErrorCode::invalid_schedule);
       auto graph = diamond();
       graph.nodes[1].after = {10};
       expect_error(prepare(graph, std::vector<NodeId>{40, 20, 10, 5}),
                    ErrorCode::invalid_schedule, 20);
     }},
    {"duplicate_schedule", [] {
       expect_error(prepare(diamond(), std::vector<NodeId>{40, 10, 10, 5}),
                    ErrorCode::invalid_schedule, 10);
     }},
    {"missing_schedule", [] {
       expect_error(prepare(diamond(), std::vector<NodeId>{40, 10, 20}),
                    ErrorCode::invalid_schedule);
       expect_error(prepare(diamond(), std::vector<NodeId>{}),
                    ErrorCode::invalid_schedule);
     }},
    {"unknown_schedule", [] {
       expect_error(prepare(diamond(), std::vector<NodeId>{40, 10, 20, 999}),
                    ErrorCode::invalid_schedule, 999);
     }},
    {"duplicate_node", [] {
       auto graph = diamond();
       graph.nodes.push_back(graph.nodes.front());
       expect_error(prepare(graph), ErrorCode::duplicate_node, 5);
     }},
    {"duplicate_slot", [] {
       auto graph = diamond();
       graph.slots.push_back({0, true});
       expect_error(prepare(graph), ErrorCode::duplicate_slot, {}, 0);
     }},
    {"unknown_input", [] {
       auto graph = diamond();
       graph.nodes.front().inputs.push_back(99);
       expect_error(prepare(graph), ErrorCode::unknown_slot, 5, 99);
     }},
    {"unknown_output", [] {
       auto graph = diamond();
       graph.nodes.front().outputs.push_back(99);
       expect_error(prepare(graph), ErrorCode::unknown_slot, 5, 99);
     }},
    {"unknown_after", [] {
       auto graph = diamond();
       graph.nodes.front().after.push_back(99);
       expect_error(prepare(graph), ErrorCode::unknown_node, 5);
     }},
    {"unknown_graph_output", [] {
       auto graph = diamond();
       graph.outputs.push_back(99);
       expect_error(prepare(graph), ErrorCode::unknown_slot, {}, 99);
     }},
    {"graph_inputs", [] {
       auto graph = diamond();
       graph.inputs.push_back(99);
       expect_error(prepare(graph), ErrorCode::unknown_slot, {}, 99);
       graph.inputs = {0, 0};
       expect_error(prepare(graph), ErrorCode::duplicate_graph_input, {}, 0);
       graph.inputs = {1};
       expect_error(prepare(graph), ErrorCode::invalid_graph_input, {}, 1);
       // Unlisted external slots can be preloaded weights/constants.
       graph.inputs.clear();
       expect_order(prepare(graph), {40, 10, 20, 5});
       // An input may be unused; listing it still defines the calling interface.
       graph.slots.push_back({99, true});
       graph.inputs = {99, 0};
       expect_order(prepare(graph), {40, 10, 20, 5});
     }},
    {"missing_producer", [] {
       auto graph = diamond();
       graph.slots.push_back({99});
       expect_error(prepare(graph), ErrorCode::missing_producer, {}, 99);
       graph.nodes.front().inputs.push_back(99);
       expect_error(prepare(graph), ErrorCode::missing_producer, 5, 99);
       expect_error(prepare(Graph{{{0}}, {}, {}, {0}}),
                    ErrorCode::missing_producer, {}, 0);
     }},
    {"multiple_producers", [] {
       auto graph = diamond();
       graph.nodes[1].outputs.push_back(4);
       expect_error(prepare(graph), ErrorCode::multiple_producers, 20, 4);
       graph = diamond();
       graph.nodes.front().outputs.push_back(4);
       expect_error(prepare(graph), ErrorCode::multiple_producers, 5, 4);
     }},
    {"external_write", [] {
       auto graph = diamond();
       graph.nodes.front().outputs.push_back(0);
       expect_error(prepare(graph), ErrorCode::external_write, 5, 0);
     }},
    {"self_dependency", [] {
       auto graph = diamond();
       graph.nodes.front().inputs.push_back(4);
       expect_error(prepare(graph), ErrorCode::self_dependency, 5, 4);
       graph = diamond();
       graph.nodes.front().after.push_back(5);
       expect_error(prepare(graph), ErrorCode::self_dependency, 5);
     }},
    {"data_cycle", [] {
       auto graph = diamond();
       graph.nodes[2].inputs = {4};
       graph.nodes.push_back({99, {}, {}, {}});  // Detect failure after progress.
       expect_error(prepare(graph), ErrorCode::cycle);
     }},
    {"after_cycle", [] {
       auto graph = diamond();
       graph.nodes[2].after = {5};
       expect_error(prepare(graph), ErrorCode::cycle);
       const Graph pure{{}, {{1, {}, {}, {2}}, {2, {}, {}, {1}}}, {}, {}};
       expect_error(prepare(pure), ErrorCode::cycle);
     }},
    {"duplicate_edges", [] {
       auto graph = diamond();
       graph.nodes.front().inputs.push_back(2);
       graph.nodes.front().after = {10, 10, 20};
       expect_order(prepare(graph), {40, 10, 20, 5});
     }},
    {"empty_graph", [] {
       expect_order(prepare(Graph{}), {});
       expect_order(prepare(Graph{}, std::vector<NodeId>{}), {});
       expect_error(prepare(Graph{}, std::vector<NodeId>{1}),
                    ErrorCode::invalid_schedule);
       expect_order(prepare(Graph{{}, {{5, {}, {}, {}}, {1, {}, {}, {}}}, {}, {}}),
                    {1, 5});
     }},
    {"external_output", [] {
       expect_order(prepare(Graph{{{0, true}}, {}, {0}, {0}}), {});
     }},
    {"zero_and_large_ids", [] {
       const auto max = std::numeric_limits<NodeId>::max();
       expect_order(prepare(Graph{{{0}, {1}},
                                  {{max, {0}, {1}, {}}, {0, {}, {0}, {}}}, {}, {1}}),
                    {0, max});
     }},
};

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 || cases.count(argv[1]) == 0) {
    std::cerr << "Expected one registered test case name\n";
    return 2;
  }
  try {
    cases.at(argv[1])();
  } catch (const std::exception& error) {
    std::cerr << argv[1] << ": " << error.what() << '\n';
    return 1;
  }
}
