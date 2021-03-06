/**
 * @file crossentropy_internal.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-30
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "tensor/tensor.h"

#if defined(MAGMADNN_HAVE_CUDA)
#include <cuda.h>
#endif

namespace magmadnn {
namespace internal {

template <typename T>
void crossentropy_full(Tensor<T> *x, Tensor<T> *y, Tensor<T> *softmax, Tensor<T> *out);

#if defined(MAGMADNN_HAVE_CUDA)
template <typename T>
void crossentropy_full_device(Tensor<T> *x, Tensor<T> *y, Tensor<T> *softmax, Tensor<T> *out);
#endif

}  // namespace internal
}  // namespace magmadnn
