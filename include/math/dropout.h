/**
 * @file dropout.h
 * @author Sedrick Keh
 * @author Florent Lopez
 * @version 0.1
 * @date 2019-06-28
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#if defined(MAGMADNN_CMAKE_BUILD)
#include "magmadnn/config.h"
#endif
#include "magmadnn/types.h"
#include "math/product.h"
#include "tensor/tensor.h"

#if defined(MAGMADNN_HAVE_CUDA)
#include "cudnn.h"
#endif

namespace magmadnn {
namespace math {

#if defined(MAGMADNN_HAVE_CUDA)
struct cudnn_dropout_settings_t {
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t xdesc;
    cudnnTensorDescriptor_t ydesc;
};

struct cudnn_dropout_grad_settings_t {
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dxdesc;
    cudnnTensorDescriptor_t dydesc;
};

struct cudnn_dropout_shared_settings_t {
    cudnnDropoutDescriptor_t dropoutDesc;
    void *states;
    void *reserveSpace;
    size_t stateSizeInBytes;
    size_t reserveSpaceSizeInBytes;
};
#endif

template <typename T>
void dropout(Tensor<T> *x, Tensor<T> *out, Tensor<T> *mask, float dropout_rate);

template <typename T>
void dropout_grad(Tensor<T> *grad, Tensor<T> *out, Tensor<T> *mask);

#if defined(MAGMADNN_HAVE_CUDA)
template <typename T>
void dropout_device(Tensor<T> *x, Tensor<T> *out, cudnn_dropout_settings_t settings,
                    cudnn_dropout_shared_settings_t shared);

template <typename T>
void dropout_grad_device(Tensor<T> *grad, Tensor<T> *out, cudnn_dropout_grad_settings_t settings,
                         cudnn_dropout_shared_settings_t shared);
#endif

}  // namespace math
}  // namespace magmadnn
