/**
 * @file softmax.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-06-12
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "tensor/tensor.h"
#include "types.h"
#include "utilities_internal.h"

#if defined(_HAS_CUDA_)
#include "cudnn.h"
#endif

namespace magmadnn {
namespace math {

#if defined(_HAS_CUDA_)
struct cudnn_softmax_settings_t {
    cudnnHandle_t handle;
    cudnnSoftmaxAlgorithm_t alg;
    cudnnSoftmaxMode_t mode;
    cudnnTensorDescriptor_t xdesc;
    cudnnTensorDescriptor_t ydesc;
};

struct cudnn_softmax_grad_settings_t {
    cudnnHandle_t handle;
    cudnnSoftmaxAlgorithm_t alg;
    cudnnSoftmaxMode_t mode;
    cudnnTensorDescriptor_t ydesc;
    cudnnTensorDescriptor_t dydesc;
    cudnnTensorDescriptor_t dxdesc;
};
#endif

template <typename T>
void softmax(Tensor<T> *x, Tensor<T> *out);

template <typename T>
void softmax_grad(Tensor<T> *softmax, Tensor<T> *grad, Tensor<T> *out);

#if defined(_HAS_CUDA_)
template <typename T>
void softmax_device(Tensor<T> *x, Tensor<T> *out, cudnn_softmax_settings_t settings);

template <typename T>
void softmax_grad_device(Tensor<T> *y, Tensor<T> *grad, Tensor<T> *out, cudnn_softmax_grad_settings_t settings);
#endif

}  // namespace math
}  // namespace magmadnn