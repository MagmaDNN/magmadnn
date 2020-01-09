/**
 * @file sigmoid_internal.h
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-02-23
 *
 * @copyright Copyright (c) 2019
 */
#pragma once
#include <math.h>
#include "tensor/tensor.h"

namespace magmadnn {
namespace internal {

template <typename T>
void sigmoid_full_cpu(Tensor<T> *x, Tensor<T> *out, bool fast);

/** Computes the element-wise sigmoid on x.
 * @tparam T
 * @param x pointer to tensor to be sigmoided
 * @param fast if true, then x=1/(1+|x|) is computed instead of normal sigmoid
 */
template <typename T>
void sigmoid_full(Tensor<T> *x, Tensor<T> *out, bool fast = true);

#if defined(MAGMADNN_HAVE_CUDA)
/** Computes the element-wise sigmoid on a device.
 * @tparam T
 * @param x tensor with device_ptr
 * @param fast
 */
template <typename T>
void sigmoid_full_device(Tensor<T> *x, Tensor<T> *out, bool fast = true);

template <typename T>
void sigmoid_full_device(
      cudaStream_t custream, Tensor<T> *x, Tensor<T> *out, bool fast = true);
#endif

template <typename T>
void sigmoid_grad_cpu(Tensor<T> *output, Tensor<T> *grad, Tensor<T> *out);

template <typename T>
void sigmoid_grad(Tensor<T> *output, Tensor<T> *grad, Tensor<T> *out);

#if defined(MAGMADNN_HAVE_CUDA)
template <typename T>
void sigmoid_grad_device(Tensor<T> *output, Tensor<T> *grad, Tensor<T> *out);

template <typename T>
void sigmoid_grad_device(
      cudaStream_t custream, Tensor<T> *output, Tensor<T> *grad, Tensor<T> *out);
#endif

}  // namespace internal
}  // namespace magmadnn
