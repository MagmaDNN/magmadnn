/**
 * @file divop.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-23
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "compute/div/div_internal.h"
#include "compute/operation.h"
#include "tensor/tensor.h"

namespace magmadnn {

namespace internal {
enum div_op_t { TENSOR_DIV_TENSOR, SCALAR_DIV_TENSOR, TENSOR_DIV_SCALAR, VEC_DIV_SCALAR, SCALAR_DIV_VEC };
}  // namespace internal

namespace op {

template <typename T>
class DivOp : public Operation<T> {
   public:
    DivOp(Operation<T> *a, Operation<T> *b, bool copy, bool needs_grad);

    std::string to_string() { return "( " + a->to_string() + " / " + b->to_string() + " )"; }

   protected:
    Tensor<T> *_eval(bool recompute = true);
    Tensor<T> &_grad(Operation<T> *consumer, Operation<T> *var, const Tensor<T> &grad);

    Operation<T> *a, *b;
    Tensor<T> *a_tensor, *b_tensor;

    internal::div_op_t op_type;

    bool copy;
};

/** Divides tensor a by b. There are 3 cases:
 *
 * 1. a and b share a shape: the element-wise division a/b is performed
 *
 * 2. a is a scalar: a->get(0) / b is performed
 *
 * 3. b is a scalar: a / b->get(0) is performed
 *
 * If a and b do not have the same shape and neither are a scalar, then it is an error.
 * @tparam T int float double
 * @param a tensor
 * @param b tensor
 * @param copy
 * @param needs_grad
 * @return DivOp<T>* the result of the above cases calculated.
 */
template <typename T>
DivOp<T> *div(Operation<T> *a, Operation<T> *b, bool copy, bool needs_grad);

}  // namespace op
}  // namespace magmadnn
