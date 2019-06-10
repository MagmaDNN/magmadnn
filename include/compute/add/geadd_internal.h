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

namespace magmadnn {
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
bool geadd_check(Tensor<T> *A, Tensor<T> *B, Tensor<T> *C);

/** Computes C = alpha*A + beta*B All tensors <i>must</i> have the same memory type and shape/size.
 * @tparam T int, float, or double
 * @param alpha 
 * @param A 
 * @param beta 
 * @param B 
 * @param C 
 */
template <typename T>
void geadd_full(T alpha, Tensor<T> *A, T beta, Tensor<T> *B, Tensor<T> *C);

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
void geadd_full_device(T alpha, Tensor<T> *A, T beta, Tensor<T> *B, Tensor<T> *C);
#endif


/**
 * @tparam T numeric
 * @param alpha 
 * @param x 
 * @param out 
 */
template <typename T>
void tensor_scalar_add_full(T alpha, Tensor<T> *x, Tensor<T> *out);

#if defined(_HAS_CUDA_)
/**
 * @tparam T numeric
 * @param alpha 
 * @param x 
 * @param out 
 */
template <typename T>
void tensor_scalar_add_full_device(T alpha, Tensor<T> *x, Tensor<T> *out);
#endif

}   // namespace internal
}   // namespace magmadnn
