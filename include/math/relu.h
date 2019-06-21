/**
 * @file relu.h
 * @author Daniel Nichols
 * @version 0.1.0
 * @date 2019-06-21
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "tensor/tensor.h"
#include "utilities_internal.h"

#if defined(_HAS_CUDA_)
#include "cudnn.h"
#endif

namespace magmadnn {
namespace math {

template <typename T>
void relu(Tensor<T> *x, Tensor<T> *out);

template <typename T>
void relu_grad(Tensor<T> *x, Tensor<T> *relu_out, Tensor<T> *grad, Tensor<T> *out);

#if defined(_HAS_CUDA_)

struct relu_cudnn_settings_t {
    cudnnActivationDescriptor_t descriptor;
};

template <typename T>
void relu_device(Tensor<T> *x, Tensor<T> *out, relu_cudnn_settings_t settings);

template <typename T>
void relu_grad_device(Tensor<T> *x, Tensor<T> *relu_out, Tensor<T> *grad, Tensor<T> *out, relu_cudnn_settings_t settings);
#endif

}
}