/**
 * @file geadd_internal.h
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-22
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once
#include "cblas.h"
#include "tensor/tensor.h"

namespace skepsi {
namespace internal {

/** Returns true if A, B, C are valid parameters to geadd_full.
 * @tparam T 
 * @param A 
 * @param B 
 * @param C 
 * @return true 
 * @return false 
 */
template <typename T>
bool geadd_check(tensor<T> *A, tensor<T> *B, tensor<T> *C);

/** Computes C = alpha*A + beta*B All tensors <i>must</i> have the same memory type and shape/size.
 * @tparam T int, float, or double
 * @param alpha 
 * @param A 
 * @param beta 
 * @param B 
 * @param C 
 */
template <typename T>
void geadd_full(T alpha, tensor<T> *A, T beta, tensor<T> *B, tensor<T> *C);

#if defined(_HAS_CUDA_)
/** Computes C = alpha*A + beta*B on gpu.
 * @tparam T 
 * @param M 
 * @param N 
 * @param alpha 
 * @param A 
 * @param beta 
 * @param B 
 * @param C 
 */
template <typename T>
void geadd_full_device(unsigned int M, unsigned int N, T alpha, T *A, T beta, T *B, T *C);
#endif

}   // namespace internal
}   // namespace skepsi
