/**
 * @file product_internal.h
 * @author Daniel Nichols
 * @author Florent Lopez
 * @version 0.1
 * @date 2019-05-21
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "tensor/tensor.h"

namespace magmadnn {
namespace internal {

template <typename T>
void product_full(T alpha, Tensor<T> *a, Tensor<T> *b, Tensor<T> *out);

#if defined(MAGMADNN_HAVE_CUDA)
template <typename T>
void product_full_device(T alpha, Tensor<T> *a, Tensor<T> *b, Tensor<T> *out);
#endif

template <typename T>
void scalar_tensor_product_full(T alpha, Tensor<T> *a, Tensor<T> *out);

#if defined(MAGMADNN_HAVE_CUDA)
template <typename T>
void scalar_tensor_product_full_device(T alpha, Tensor<T> *a, Tensor<T> *out);
#endif

}  // namespace internal
}  // namespace magmadnn
