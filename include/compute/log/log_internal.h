
#pragma once

#include "magmadnn/config.h"
#include "tensor/tensor.h"

#include <cmath>

namespace magmadnn {
namespace internal {

template <typename T>
void log_full_cpu(Tensor<T> *x, Tensor<T> *out, bool stable = false);

template <typename T>
void log_full(Tensor<T> *x, Tensor<T> *out, bool stable = false);

#if defined(MAGMADNN_HAVE_CUDA)
template <typename T>
void log_full_device(Tensor<T> *x, Tensor<T> *out, bool stable = false);

template <typename T>
void log_full_device(cudaStream_t custream, Tensor<T> *x, Tensor<T> *out, bool stable = false);
#endif

template <typename T>
void log_grad_cpu(Tensor<T> *x, Tensor<T> *grad, Tensor<T> *out, bool stable = false);

template <typename T>
void log_grad(Tensor<T> *x, Tensor<T> *grad, Tensor<T> *out, bool stable = false);

#if defined(MAGMADNN_HAVE_CUDA)
template <typename T>
void log_grad_device(Tensor<T> *x, Tensor<T> *grad, Tensor<T> *out, bool stable = false);

template <typename T>
void log_grad_device(cudaStream_t custream, Tensor<T> *x, Tensor<T> *grad, Tensor<T> *out, bool stable = false);
#endif

}  // namespace internal
}  // namespace magmadnn
