/**
 * @file relu.h
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-06-21
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "magmadnn/config.h"
#include "tensor/tensor.h"

#if defined(MAGMADNN_HAVE_CUDA)
#include "cudnn.h"
#endif

namespace magmadnn {
namespace math {

template <typename T>
void relu(Tensor<T> *x, Tensor<T> *out);

template <typename T>
void relu_grad(Tensor<T> *x, Tensor<T> *relu_out, Tensor<T> *grad, Tensor<T> *out);

#if defined(MAGMADNN_HAVE_CUDA)

struct relu_cudnn_settings_t {
    cudnnActivationDescriptor_t descriptor;
    cudnnHandle_t handle;
};

template <typename T>
void relu_device(Tensor<T> *x, Tensor<T> *out, relu_cudnn_settings_t settings);

template <typename T>
void relu_grad_device(Tensor<T> *x, Tensor<T> *relu_out, Tensor<T> *grad, Tensor<T> *out,
                      relu_cudnn_settings_t settings);
#endif

}  // namespace math
}  // namespace magmadnn
