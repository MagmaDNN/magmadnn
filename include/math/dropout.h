/**
 * @file dropout.h
 * @author Sedrick Keh
 * @version 0.1
 * @date 2019-06-28
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "math/binary_math_operations.h"
#include "math/launch_math_kernel.h"
#include "mdnn_device_types.h"
#include "tensor/tensor.h"
#include "types.h"
#include "utilities_internal.h"

#if defined(_HAS_CUDA_)
#include "cudnn.h"
#endif

namespace magmadnn {
namespace math {

#if defined(_HAS_CUDA_)
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
void dropout(const Tensor &x, Tensor &out, const Tensor &mask, float dropout_rate);

template <typename T>
void dropout_grad(const Tensor &grad, Tensor &out, const Tensor &mask);

#if defined(_HAS_CUDA_)
template <typename T>
void dropout_device(const Tensor &x, Tensor &out, cudnn_dropout_settings_t settings,
                    cudnn_dropout_shared_settings_t shared);

template <typename T>
void dropout_grad_device(const Tensor &grad, Tensor &out, cudnn_dropout_grad_settings_t settings,
                         cudnn_dropout_shared_settings_t shared);
#endif

}  // namespace math
}  // namespace magmadnn