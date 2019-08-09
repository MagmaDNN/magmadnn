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
#include "utilities_internal.h"

namespace magmadnn {
namespace op {

class Graph {
   public:
    template <typename op_type, typename... Args>
    inline Operation* add_operation(Args... args);

    /* TODO -- add graph statistics and manipulation functions */

   protected:
    std::vector<std::unique_ptr<Operation>> nodes; /* operations in the graph */
};

template <typename op_type, typename... Args>
inline Operation* Graph::add_operation(Args... args) {
    // std::unique_ptr<Operation> tmp_ptr{new op_type(args)};
    std::unique_ptr<Operation> tmp_ptr = ::magmadnn::internal::make_unique<op_type>(args...);

    /* use std::move to transfer ownership */
    this->nodes.push_back(std::move(tmp_ptr));

    /*  Return the raw pointer
        This is ok since only the graph will have owenership of the operations.
        We want pointers, because operations use them _algorithmically_ for graph operations
     */
    return nodes.back().get();
}

Graph default_graph;

}  // namespace op
}  // namespace magmadnn