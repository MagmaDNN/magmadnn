
#pragma once

#include "tensor/tensor.h"

namespace magmadnn {
namespace internal {

template <typename T>
void scalarproduct_full(T alpha, Tensor<T> *x, Tensor<T> *out);

#if defined(MAGMADNN_HAVE_CUDA)
template <typename T>
void scalarproduct_full_device(T alpha, Tensor<T> *x, Tensor<T> *out);
#endif

}  // namespace internal
}  // namespace magmadnn
