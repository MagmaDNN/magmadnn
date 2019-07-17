/**
 * @file compute_graph.h
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-07-17
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include <memory>
#include <vector>

#include "compute/operation.h"

namespace magmadnn {
namespace op {

template <typename T>
class Graph {
   public:
    template <template <typename M> typename op_type, typename... Args>
    Operation<T>* add_operation(Args... args);

   protected:
    std::vector<std::unique_ptr<Operation<T>>> nodes; /* operations in the graph */
};

Graph<int> DEFAULT_I_GRAPH;
Graph<float> DEFAULT_F_GRAPH;
Graph<double> DEFAULT_D_GRAPH;

template <typename T>
Graph<T>& get_graph();

}  // namespace op
}  // namespace magmadnn