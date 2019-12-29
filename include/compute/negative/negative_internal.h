
#pragma once

#include "tensor/tensor.h"

namespace magmadnn {
namespace internal {

template <typename T>
void negative_full(Tensor<T> *x, Tensor<T> *out);

template <typename T>
void negative_full_cpu(Tensor<T> *x, Tensor<T> *out);
   
#if defined(MAGMADNN_HAVE_CUDA)
template <typename T>
void negative_full_device(Tensor<T> *x, Tensor<T> *out);

template <typename T>
void negative_full_device(cudaStream_t custream, Tensor<T> *x, Tensor<T> *out);
#endif

}  // namespace internal
}  // namespace magmadnn
