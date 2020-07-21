/**
 * @file batchnorm.cpp
 * @author Sedrick Keh
 * @author Florent Lopez
 * @version 1.0
 * @date 2019-07-24
 *
 * @copyright Copyright (c) 2019
 */
#include "math/batchnorm.h"

#include <cassert>

#if defined(MAGMADNN_CMAKE_BUILD)
#include "magmadnn/config.h"
#endif

namespace magmadnn {
namespace math {

template <typename T>
void batchnorm(Tensor<T> *x, Tensor<T> *out) {
    assert(T_IS_SAME_MEMORY_TYPE(x, out));

    if (out->get_memory_type() == HOST) {
        fprintf(stderr, "batchnorm CPU not supported yet.\n");
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {
        fprintf(stderr, "For batchnorm on GPU, please use batchnorm_device\n");
    }
#endif
}
template void batchnorm(Tensor<int> *x, Tensor<int> *out);
template void batchnorm(Tensor<float> *x, Tensor<float> *out);
template void batchnorm(Tensor<double> *x, Tensor<double> *out);

template <typename T>
void batchnorm_grad(Tensor<T> *grad, Tensor<T> *out) {
    assert(T_IS_SAME_MEMORY_TYPE(grad, out));

    if (out->get_memory_type() == HOST) {
        fprintf(stderr, "batchnorm grad CPU not supported yet.\n");
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {
        fprintf(stderr, "For batchnorm_grad on GPU, please use batchnorm_grad_device\n");
    }
#endif
}
template void batchnorm_grad(Tensor<int> *grad, Tensor<int> *out);
template void batchnorm_grad(Tensor<float> *grad, Tensor<float> *out);
template void batchnorm_grad(Tensor<double> *grad, Tensor<double> *out);

#if defined(MAGMADNN_HAVE_CUDA)
template <typename T>
void batchnorm_device(Tensor<T> *x, Tensor<T> *out, Tensor<T> *bn_scale, Tensor<T> *bn_bias, Tensor<T> *running_mean,
                      Tensor<T> *running_variance, Tensor<T> *saved_mean, Tensor<T> *saved_variance,
                      unsigned int &num_calls, cudnn_batchnorm_settings_t settings) {
    T alpha = static_cast<T>(1), beta = static_cast<T>(0);
    double epsilon = 1E-8;
    num_calls++;

    cudnnErrchk(cudnnBatchNormalizationForwardTraining(
        settings.handle, settings.mode, &alpha, &beta, x->get_cudnn_tensor_descriptor(), x->get_ptr(),
        out->get_cudnn_tensor_descriptor(), out->get_ptr(), settings.bn_tensor_desc, bn_scale->get_ptr(),
        bn_bias->get_ptr(), ((double) (1) / (double) (1 + num_calls)), running_mean->get_ptr(),
        running_variance->get_ptr(), epsilon, saved_mean->get_ptr(), saved_variance->get_ptr()));
}
template void batchnorm_device(Tensor<int> *x, Tensor<int> *out, Tensor<int> *bn_scale, Tensor<int> *bn_bias,
                               Tensor<int> *running_mean, Tensor<int> *running_variance, Tensor<int> *saved_mean,
                               Tensor<int> *saved_variance, unsigned int &num_calls,
                               cudnn_batchnorm_settings_t settings);
template void batchnorm_device(Tensor<float> *x, Tensor<float> *out, Tensor<float> *bn_scale, Tensor<float> *bn_bias,
                               Tensor<float> *running_mean, Tensor<float> *running_variance, Tensor<float> *saved_mean,
                               Tensor<float> *saved_variance, unsigned int &num_calls,
                               cudnn_batchnorm_settings_t settings);
template void batchnorm_device(Tensor<double> *x, Tensor<double> *out, Tensor<double> *bn_scale,
                               Tensor<double> *bn_bias, Tensor<double> *running_mean, Tensor<double> *running_variance,
                               Tensor<double> *saved_mean, Tensor<double> *saved_variance, unsigned int &num_calls,
                               cudnn_batchnorm_settings_t settings);

template <typename T>
void batchnorm_grad_device(Tensor<T> *x, Tensor<T> *grad, Tensor<T> *out, Tensor<T> *bn_scale, Tensor<T> *bn_scale_diff,
                           Tensor<T> *bn_bias_diff, Tensor<T> *saved_mean, Tensor<T> *saved_variance,
                           cudnn_batchnorm_settings_t settings) {
    T alpha_data = static_cast<T>(1), alpha_params = static_cast<T>(1);
    T beta_data = static_cast<T>(0), beta_params = static_cast<T>(0);
    double epsilon = 1E-8;

    cudnnErrchk(cudnnBatchNormalizationBackward(
        settings.handle, settings.mode, &alpha_data, &beta_data, &alpha_params, &beta_params,
        x->get_cudnn_tensor_descriptor(), x->get_ptr(), grad->get_cudnn_tensor_descriptor(), grad->get_ptr(),
        out->get_cudnn_tensor_descriptor(), out->get_ptr(), settings.bn_tensor_desc, bn_scale->get_ptr(),
        bn_scale_diff->get_ptr(), bn_bias_diff->get_ptr(), epsilon, saved_mean->get_ptr(), saved_variance->get_ptr()));
}
template void batchnorm_grad_device(Tensor<int> *x, Tensor<int> *grad, Tensor<int> *out, Tensor<int> *bn_scale,
                                    Tensor<int> *bn_scale_diff, Tensor<int> *bn_bias_diff, Tensor<int> *saved_mean,
                                    Tensor<int> *saved_variance, cudnn_batchnorm_settings_t settings);
template void batchnorm_grad_device(Tensor<float> *x, Tensor<float> *grad, Tensor<float> *out, Tensor<float> *bn_scale,
                                    Tensor<float> *bn_scale_diff, Tensor<float> *bn_bias_diff,
                                    Tensor<float> *saved_mean, Tensor<float> *saved_variance,
                                    cudnn_batchnorm_settings_t settings);
template void batchnorm_grad_device(Tensor<double> *x, Tensor<double> *grad, Tensor<double> *out,
                                    Tensor<double> *bn_scale, Tensor<double> *bn_scale_diff,
                                    Tensor<double> *bn_bias_diff, Tensor<double> *saved_mean,
                                    Tensor<double> *saved_variance, cudnn_batchnorm_settings_t settings);

#endif

}  // namespace math
}  // namespace magmadnn
