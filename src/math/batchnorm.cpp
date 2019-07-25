/**
 * @file batchnorm.cpp
 * @author Sedrick Keh
 * @version 1.0
 * @date 2019-07-24
 *
 * @copyright Copyright (c) 2019
 */
#include "math/batchnorm.h"

namespace magmadnn {
namespace math {

template <typename T>
void batchnorm(const Tensor &x, Tensor &out) {
    // assert(T_IS_SAME_MEMORY_TYPE(x, out));

    if (out.get_memory_type() == HOST) {
        LOG(ERROR) << "batchnorm CPU not supported yet.\n";
    }
#if defined(_HAS_CUDA_)
    else {
        LOG(ERROR) << "For batchnorm on GPU, please use batchnorm_device\n";
    }
#endif
}
#define comp(type) template void batchnorm<type>(const Tensor &, Tensor &);
CALL_FOR_ALL_TYPES(comp)
#undef comp

template <typename T>
void batchnorm_grad(const Tensor &grad, Tensor &out) {
    // assert(T_IS_SAME_MEMORY_TYPE(grad, out));

    if (out.get_memory_type() == HOST) {
        LOG(ERROR) << "batchnorm grad CPU not supported yet.\n";
    }
#if defined(_HAS_CUDA_)
    else {
        LOG(ERROR) << "For batchnorm_grad on GPU, please use batchnorm_grad_device\n";
    }
#endif
}
#define comp(type) template void batchnorm_grad<type>(const Tensor &, Tensor &);
CALL_FOR_ALL_TYPES(comp)
#undef comp

#if defined(_HAS_CUDA_)
template <typename T>
void batchnorm_device(const Tensor &x, Tensor &out, Tensor &bn_scale, Tensor &bn_bias, Tensor &running_mean,
                      Tensor &running_variance, Tensor &saved_mean, Tensor &saved_variance, unsigned int &num_calls,
                      cudnn_batchnorm_settings_t settings) {
    T alpha = static_cast<T>(1), beta = static_cast<T>(0);
    double epsilon = 1E-8;
    num_calls++;

    cudnnErrchk(cudnnBatchNormalizationForwardTraining(
        settings.handle, settings.mode, &alpha, &beta, x.get_cudnn_tensor_descriptor(), x.get_ptr<T>(),
        out.get_cudnn_tensor_descriptor(), out.get_ptr<T>(), settings.bn_tensor_desc, bn_scale.get_ptr<T>(),
        bn_bias.get_ptr<T>(), ((double) (1) / (double) (1 + num_calls)), running_mean.get_ptr<T>(),
        running_variance.get_ptr<T>(), epsilon, saved_mean.get_ptr<T>(), saved_variance.get_ptr<T>()));
}
#define comp(type)                                                                                                   \
    template void batchnorm_device<type>(const Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, \
                                         Tensor &, unsigned int &num_calls, cudnn_batchnorm_settings_t settings);
CALL_FOR_ALL_TYPES(comp)
#undef comp

template <typename T>
void batchnorm_grad_device(const Tensor &x, const Tensor &grad, Tensor &out, Tensor &bn_scale, Tensor &bn_scale_diff,
                           Tensor &bn_bias_diff, Tensor &saved_mean, Tensor &saved_variance,
                           cudnn_batchnorm_settings_t settings) {
    T alpha_data = static_cast<T>(1), alpha_params = static_cast<T>(1);
    T beta_data = static_cast<T>(0), beta_params = static_cast<T>(0);
    double epsilon = 1E-8;

    cudnnErrchk(cudnnBatchNormalizationBackward(
        settings.handle, settings.mode, &alpha_data, &beta_data, &alpha_params, &beta_params,
        x.get_cudnn_tensor_descriptor(), x.get_ptr<T>(), grad.get_cudnn_tensor_descriptor(), grad.get_ptr<T>(),
        out.get_cudnn_tensor_descriptor(), out.get_ptr<T>(), settings.bn_tensor_desc, bn_scale.get_ptr<T>(),
        bn_scale_diff.get_ptr<T>(), bn_bias_diff.get_ptr<T>(), epsilon, saved_mean.get_ptr<T>(),
        saved_variance.get_ptr<T>()));
}
#define comp(type)                                                                                                    \
    template void batchnorm_grad_device<type>(const Tensor &, const Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, \
                                              Tensor &, Tensor &, cudnn_batchnorm_settings_t settings);
CALL_FOR_ALL_TYPES(comp)
#undef comp

#endif

}  // namespace math
}  // namespace magmadnn