/**
 * @file geadd_internal.h
 * @author Daniel Nichols
 * @author Florent Lopez
 * @version 1.0
 * @date 2019-02-22
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "magmadnn/config.h"
#include "tensor/tensor.h"

namespace magmadnn {
namespace internal {

/** Returns true if A, B, C are valid parameters to geadd_full.
 * @tparam T
 * @param A a tensor
 * @param B a tensor
 * @param C a tensor
 * @return true
 * @return false
 */
template <typename T>
bool geadd_check(Tensor<T> *A, Tensor<T> *B, Tensor<T> *C);

template <typename T>
void geadd_full_cpu(T alpha, Tensor<T> *A, T beta, Tensor<T> *B, Tensor<T> *C);

/** Computes C = alpha*A + beta*B All tensors <i>must</i> have the same memory type and shape/size.
 * @tparam T int, float, or double
 * @param alpha scaling value
 * @param A a tensor
 * @param beta scaling value
 * @param B a tensor
 * @param C a tensor
 */
template <typename T>
void geadd_full(T alpha, Tensor<T> *A, T beta, Tensor<T> *B, Tensor<T> *C);

#if defined(MAGMADNN_HAVE_CUDA)
/** Computes C=alpha*A + beta*B
 * @tparam T int, float, double
 * @param alpha scaling value
 * @param A tensor
 * @param beta scaling value
 * @param B tensor
 * @param C tensor
 */
template <typename T>
void geadd_full_device(T alpha, Tensor<T> *A, T beta, Tensor<T> *B, Tensor<T> *C);

template <typename T>
void geadd_full_device(cudaStream_t custream, T alpha, Tensor<T> *A, T beta, Tensor<T> *B, Tensor<T> *C);
#endif

template <typename T>
void tensor_scalar_add_full_cpu(T alpha, Tensor<T> *x, Tensor<T> *out);

/**
 * @tparam T numeric
 * @param alpha
 * @param x
 * @param out
 */
template <typename T>
void tensor_scalar_add_full(T alpha, Tensor<T> *x, Tensor<T> *out);

#if defined(MAGMADNN_HAVE_CUDA)
/**
 * @tparam T numeric
 * @param alpha
 * @param x
 * @param out
 */
template <typename T>
void tensor_scalar_add_full_device(T alpha, Tensor<T> *x, Tensor<T> *out);

template <typename T>
void tensor_scalar_add_full_device(cudaStream_t custream, T alpha, Tensor<T> *x, Tensor<T> *out);
#endif

}  // namespace internal
}  // namespace magmadnn
