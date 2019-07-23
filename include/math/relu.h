/**
 * @file relu.h
 * @author Daniel Nichols
 * @version 1.0
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
void relu(const Tensor &x, Tensor &out);

template <typename T>
void relu_grad(const Tensor &x, const Tensor &relu_out, const Tensor &grad, Tensor &out);

#if defined(_HAS_CUDA_)

struct relu_cudnn_settings_t {
    cudnnActivationDescriptor_t descriptor;
};

template <typename T>
void relu_device(const Tensor &x, Tensor &out, relu_cudnn_settings_t settings);

template <typename T>
void relu_grad_device(const Tensor &x, const Tensor &relu_out, const Tensor &grad, Tensor &out,
                      relu_cudnn_settings_t settings);
#endif

}  // namespace math
}  // namespace magmadnn