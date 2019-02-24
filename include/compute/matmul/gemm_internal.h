/**
 * @file gemm_internal.h
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-22
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once
#include "cblas.h"
#include "tensor/tensor.h"

#if defined(_HAS_CUDA_)
#include "magma.h"
#endif

namespace skepsi {
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
bool gemm_check(tensor<T> *A, tensor<T> *B, tensor<T> *C, unsigned int &M, unsigned int &N, unsigned int &K);

/** Computes the matrix product C = alpha*(AB) + beta*C
 * @tparam T 
 * @param alpha 
 * @param A 
 * @param B 
 * @param beta 
 * @param C 
 */
template <typename T>
void gemm_full(T alpha, tensor<T>* A, tensor<T>* B, T beta, tensor<T>* C);


}   // namespace internal
}   // namespace skepsi
