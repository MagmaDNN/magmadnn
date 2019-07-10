/**
 * @file matmulop.h
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-02-20
 *
 * @copyright Copyright (c) 2019
 */
#pragma once
#include <vector>
#include "compute/dot/dotop.h"
#include "compute/operation.h"
#include "compute/transpose/transposeop.h"
#include "compute/variable.h"
#include "gemm_internal.h"
#include "math/dot.h"
#include "math/matmul.h"
#include "tensor/tensor.h"

namespace magmadnn {
namespace op {

template <typename T>
class MatmulOp : public Operation<T> {
   public:
    MatmulOp(T alpha, Operation<T> *a, Operation<T> *b, T beta, Operation<T> *c, bool copy = true,
             bool needs_grad = true);

    std::string to_string() { return "(" + a->to_string() + " x " + b->to_string() + ")"; }

   protected:
    Tensor<T> *_eval(bool recompute = true);
    Tensor<T> *_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad);

    Operation<T> *a;
    Operation<T> *b;
    Operation<T> *c;

    Tensor<T> *a_tensor;
    Tensor<T> *b_tensor;
    Tensor<T> *c_tensor;

    T alpha;
    T beta;
    bool copy;
};

/** Returns a new operation of type matmul. It computes the matrix product of A and B.
 * @tparam T
 * @param a
 * @param b
 * @param copy
 * @return MatmulOp<T>*
 */
template <typename T>
MatmulOp<T> *matmul(Operation<T> *a, Operation<T> *b, bool needs_grad = true);

/** Computes the full gemm C = alpha*(AB) + beta*(C). Overwrites C and returns it if copy is false. If true,
 * 	then it returns a copy of C.
 * @tparam T
 * @param alpha
 * @param a
 * @param b
 * @param beta
 * @param c
 * @param copy
 * @return MatmulOp<T>*
 */
template <typename T>
MatmulOp<T> *matmul(T alpha, Operation<T> *a, Operation<T> *b, T beta, Tensor<T> *c, bool copy = true,
                    bool needs_grad = true);

}  // namespace op
}  // namespace magmadnn
