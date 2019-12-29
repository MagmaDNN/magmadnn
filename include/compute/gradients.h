/**
 * @file gradients.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-17
 *
 * @copyright Copyright (c) 2019
 */

#pragma once

#include <vector>
#include "compute/add/addop.h"
#include "compute/gradtable.h"
#include "compute/operation.h"
#include "compute/sum/sumop.h"
#include "compute/variable.h"

namespace magmadnn {
namespace op {

/** Given a list of vars and compute graph, fills in a GradTable.
 * @tparam T numeric
 * @param vars A list of variables whose gradients will be computed
 * @param graph Head node of compute graph that contains 'vars'
 * @param table GradTable to be filled in
 * @return magmadnn_error_t non-zero on error
 */
template <typename T>
magmadnn_error_t get_grad_table(const std::vector<Operation<T> *> &vars, Operation<T> *graph, GradTable<T> &table);

}  // namespace op

// build_grad should only be used internally
namespace internal {

/** Sets the gradients for var.
 * @tparam T numeric
 * @param var Variable to compute gradients for
 * @param graph Compute graph that contains var
 * @param table GradTable to put gradients in
 * @return magmadnn_error_t non-zero on error
 */
template <typename T>
magmadnn_error_t build_grad(op::Operation<T> *var, op::Operation<T> *graph, op::GradTable<T> &table, Tensor<T> **grad);

}  // namespace internal
}  // namespace magmadnn
