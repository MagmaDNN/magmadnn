
#pragma once

#if defined(MAGMADNN_CMAKE_BUILD)
#include "magmadnn/config.h"
#endif
#include "tensor/tensor.h"

namespace magmadnn {
namespace internal {

template <typename T>
void scalarproduct_full_cpu(T alpha, Tensor<T> *x, Tensor<T> *out);
   
template <typename T>
void scalarproduct_full(T alpha, Tensor<T> *x, Tensor<T> *out);

#if defined(MAGMADNN_HAVE_CUDA)
template <typename T>
void scalarproduct_full_device(T alpha, Tensor<T> *x, Tensor<T> *out);

template <typename T>
void scalarproduct_full_device(cudaStream_t custream, T alpha, Tensor<T> *x, Tensor<T> *out);
#endif

}  // namespace internal
}  // namespace magmadnn
