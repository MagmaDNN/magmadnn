/**
 * @file batchnorm.h
 * @author Sedrick Keh
 * @version 1.0
 * @date 2019-07-24
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

template <typename T>
void batchnorm(const Tensor &x, Tensor &out);

template <typename T>
void batchnorm_grad(const Tensor &grad, Tensor &out);

#if defined(_HAS_CUDA_)

struct cudnn_batchnorm_settings_t {
    cudnnHandle_t handle;
    cudnnBatchNormMode_t mode;
    cudnnTensorDescriptor_t bn_tensor_desc;
};

template <typename T>
void batchnorm_device(const Tensor &x, Tensor &out, Tensor &bn_scale, Tensor &bn_bias, Tensor &running_mean,
                      Tensor &running_variance, Tensor &saved_mean, Tensor &saved_variance, unsigned int &num_calls,
                      cudnn_batchnorm_settings_t settings);
template <typename T>
void batchnorm_grad_device(const Tensor &x, const Tensor &grad, Tensor &out, Tensor &bn_scale, Tensor &bn_scale_diff,
                           Tensor &bn_bias_diff, Tensor &saved_mean, Tensor &saved_variance,
                           cudnn_batchnorm_settings_t settings);

#endif

}  // namespace math
}  // namespace magmadnn