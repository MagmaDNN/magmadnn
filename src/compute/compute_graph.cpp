/**
 * @file compute_graph.cpp
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-07-17
 *
 * @copyright Copyright (c) 2019
 */
#include "compute/compute_graph.h"

namespace magmadnn {
namespace op {

template <typename T>
template <template <typename M> typename op_type, typename... Args>
Operation<T>* Graph<T>::add_operation(Args... args) {
    std::unique_ptr<Operation<T>> tmp_ptr{new op_type<T>(args)};

    /* use std::move to transfer ownership */
    this->nodes.push_back(std::move(tmp_ptr));

    return tmp_ptr.get();
}

template <>
Graph<int>& get_graph() {
    return DEFAULT_I_GRAPH;
}

template <>
Graph<float>& get_graph() {
    return DEFAULT_F_GRAPH;
}

template <>
Graph<double>& get_graph() {
    return DEFAULT_D_GRAPH;
}

}  // namespace op
}  // namespace magmadnn