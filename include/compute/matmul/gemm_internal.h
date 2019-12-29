/**
 * @file gemm_internal.h
 * @author Daniel Nichols
 * @author Florent Lopez
 * @version 1.0
 * @date 2019-02-22
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "tensor/tensor.h"

#if defined(MAGMADNN_HAVE_CUDA)
#include "magma.h"
#endif

namespace magmadnn {
namespace internal {

/** Returns true if A, B, C are valid parameters for gemm_full. It also sets M, N, K to
 *  A.get_shape(0), B.get_shape(1), and A.get_shape(1), respectively.
 * @tparam T
 * @param A
 * @param B
 * @param C
 * @param M
 * @param N
 * @param K
 * @return true
 * @return false
 */
template <typename T>
bool gemm_check(Tensor<T> *A, Tensor<T> *B, Tensor<T> *C, /*unsigned*/ int &M, /*unsigned*/ int &N, /*unsigned*/ int &K);

/** Computes the matrix product C = alpha*(AB) + beta*C
 * @tparam T
 * @param alpha
 * @param A
 * @param B
 * @param beta
 * @param C
 */
template <typename T>
void gemm_full(T alpha, Tensor<T> *A, Tensor<T> *B, T beta, Tensor<T> *C);

}  // namespace internal
}  // namespace magmadnn
