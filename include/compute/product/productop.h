/**
 * @file productop.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-21
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include <vector>
#include "compute/operation.h"
#include "compute/product/product_internal.h"
#include "compute/variable.h"
#include "math/scalar_tensor_product.h"
#include "tensor/tensor.h"
#include "utilities_internal.h"

namespace magmadnn {

namespace internal {
enum product_op_t { SCALAR_PROD_TENSOR, TENSOR_PROD_SCALAR, TENSOR_PROD_TENSOR };
}  // namespace internal

namespace op {

template <typename T>
class ProductOp : public Operation<T> {
   public:
    ProductOp(T alpha, Operation<T> *a, Operation<T> *b, bool copy = true, bool needs_grad = true);

    std::string to_string() { return "(" + a->to_string() + " * " + b->to_string() + ")"; }

   protected:
    Tensor<T> *_eval(bool recompute = true);
    Tensor<T> *_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad);

    T alpha;
    Operation<T> *a;
    Operation<T> *b;

    Tensor<T> *a_tensor;
    Tensor<T> *b_tensor;

    internal::product_op_t op_type;

    bool copy;
};

template <typename T>
ProductOp<T> *product(Operation<T> *a, Operation<T> *b, bool copy = true, bool needs_grad = true);

template <typename T>
ProductOp<T> *product(T alpha, Operation<T> *a, Operation<T> *b, bool copy = true, bool needs_grad = true);

}  // namespace op
}  // namespace magmadnn