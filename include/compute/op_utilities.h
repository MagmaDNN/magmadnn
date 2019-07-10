/**
 * @file op_utilities.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-06-19
 *
 * @copyright Copyright (c) 2019
 */
#pragma once
#include <deque>
#include <set>
#include "compute/operation.h"
#include "utilities_internal.h"

namespace magmadnn {
namespace op {
namespace utility {

template <typename T>
magmadnn_error_t print_compute_graph(::magmadnn::op::Operation<T> *_root, bool debug = true);
}
}  // namespace op
}  // namespace magmadnn