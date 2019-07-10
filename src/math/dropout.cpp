/**
 * @file dropout.cpp
 * @author Sedrick Keh
 * @version 0.1
 * @date 2019-07-01
 *
 * @copyright Copyright (c) 2019
 */
#include "math/dropout.h"

namespace magmadnn {
namespace math {

template <typename T>
void dropout(Tensor<T> *x, Tensor<T> *out, Tensor<T> *mask, float dropout_rate) {
    if (out->get_memory_type() == HOST) {
        float p = 1.0f - dropout_rate;
        Tensor<T> a(mask->get_shape(), {MASK, {static_cast<T>(p), static_cast<T>(1.0f / p)}}, mask->get_memory_type());
        mask->copy_from(a);
        math::product(mask, x, out);
    } else {
        fprintf(stderr, "For dropout on GPU, please use dropout_device\n");
    }
}
template void dropout(Tensor<int> *x, Tensor<int> *out, Tensor<int> *mask, float dropout_rate);
template void dropout(Tensor<float> *x, Tensor<float> *out, Tensor<float> *mask, float dropout_rate);
template void dropout(Tensor<double> *x, Tensor<double> *out, Tensor<double> *mask, float dropout_rate);

template <typename T>
void dropout_grad(Tensor<T> *grad, Tensor<T> *out, Tensor<T> *mask) {
    if (out->get_memory_type() == HOST) {
        math::product(mask, grad, out);
    }
#if defined(_HAS_CUDA_)
    else {
        fprintf(stderr, "For dropout_grad on GPU, please use dropout_grad_device\n");
    }
#endif
}
template void dropout_grad(Tensor<int> *grad, Tensor<int> *out, Tensor<int> *mask);
template void dropout_grad(Tensor<float> *grad, Tensor<float> *out, Tensor<float> *mask);
template void dropout_grad(Tensor<double> *grad, Tensor<double> *out, Tensor<double> *mask);

#if defined(_HAS_CUDA_)
template <typename T>
void dropout_device(Tensor<T> *x, Tensor<T> *out, cudnn_dropout_settings_t settings,
                    cudnn_dropout_shared_settings_t shared) {
    cudnnErrchk(cudnnDropoutForward(settings.handle, shared.dropoutDesc, settings.xdesc, (void *) x->get_ptr(),
                                    settings.ydesc, (void *) out->get_ptr(), shared.reserveSpace,
                                    shared.reserveSpaceSizeInBytes));
}
template void dropout_device(Tensor<int> *x, Tensor<int> *out, cudnn_dropout_settings_t settings,
                             cudnn_dropout_shared_settings_t shared);
template void dropout_device(Tensor<float> *x, Tensor<float> *out, cudnn_dropout_settings_t settings,
                             cudnn_dropout_shared_settings_t shared);
template void dropout_device(Tensor<double> *x, Tensor<double> *out, cudnn_dropout_settings_t settings,
                             cudnn_dropout_shared_settings_t shared);

template <typename T>
void dropout_grad_device(Tensor<T> *grad, Tensor<T> *out, cudnn_dropout_grad_settings_t settings,
                         cudnn_dropout_shared_settings_t shared) {
    cudnnErrchk(cudnnDropoutBackward(settings.handle, shared.dropoutDesc, settings.dydesc, (void *) grad->get_ptr(),
                                     settings.dxdesc, (void *) out->get_ptr(), shared.reserveSpace,
                                     shared.reserveSpaceSizeInBytes));
}
template void dropout_grad_device(Tensor<int> *grad, Tensor<int> *out, cudnn_dropout_grad_settings_t settings,
                                  cudnn_dropout_shared_settings_t shared);
template void dropout_grad_device(Tensor<float> *grad, Tensor<float> *out, cudnn_dropout_grad_settings_t settings,
                                  cudnn_dropout_shared_settings_t shared);
template void dropout_grad_device(Tensor<double> *grad, Tensor<double> *out, cudnn_dropout_grad_settings_t settings,
                                  cudnn_dropout_shared_settings_t shared);

#endif

}  // namespace math
}  // namespace magmadnn