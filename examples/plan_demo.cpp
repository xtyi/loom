#include "plan/graph.h"

#include <iostream>

int main() {
  loom::plan::Graph graph;
  const auto input = graph.add_input(0);
  const auto hidden = graph.add_slot(1);
  const auto left = graph.add_slot(2);
  const auto right = graph.add_slot(3);
  const auto output = graph.add_slot(4);

  // 刻意按非执行顺序添加节点：顺序由数据依赖决定，不由添加顺序或 id 决定。
  graph.add_node(5).reads({left, right}).writes({output});
  graph.add_node(20).reads({hidden}).writes({right});
  graph.add_node(40).reads({input}).writes({hidden});
  graph.add_node(10).reads({hidden}).writes({left});
  graph.set_outputs({output});
  const auto result = loom::plan::prepare(graph);
  if (!result.ok()) {
    std::cerr << result.error->message << '\n';
    return 1;
  }
  std::cout << "Prepared order (no kernels executed):";
  for (const auto id : result.order) {
    std::cout << ' ' << id;
  }
  std::cout << '\n';
}
