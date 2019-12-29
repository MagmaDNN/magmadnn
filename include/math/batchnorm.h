/**
 * @file batchnorm.h
 * @author Sedrick Keh
 * @author Florent Lopez
 * @version 1.0
 * @date 2019-07-24
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "magmadnn/types.h"
#include "magmadnn/utilities_internal.h"
#include "tensor/tensor.h"

#if defined(MAGMADNN_HAVE_CUDA)
#include "cudnn.h"
#endif

namespace magmadnn {
namespace math {

template <typename T>
void batchnorm(Tensor<T> *x, Tensor<T> *out);

template <typename T>
void batchnorm_grad(Tensor<T> *grad, Tensor<T> *out);

#if defined(MAGMADNN_HAVE_CUDA)

struct cudnn_batchnorm_settings_t {
    cudnnHandle_t handle;
    cudnnBatchNormMode_t mode;
    cudnnTensorDescriptor_t bn_tensor_desc;
};

template <typename T>
void batchnorm_device(Tensor<T> *x, Tensor<T> *out, Tensor<T> *bn_scale, Tensor<T> *bn_bias, Tensor<T> *running_mean,
                      Tensor<T> *running_variance, Tensor<T> *saved_mean, Tensor<T> *saved_variance,
                      unsigned int &num_calls, cudnn_batchnorm_settings_t settings);
template <typename T>
void batchnorm_grad_device(Tensor<T> *x, Tensor<T> *grad, Tensor<T> *out, Tensor<T> *bn_scale, Tensor<T> *bn_scale_diff,
                           Tensor<T> *bn_bias_diff, Tensor<T> *saved_mean, Tensor<T> *saved_variance,
                           cudnn_batchnorm_settings_t settings);

#endif

}  // namespace math
}  // namespace magmadnn
