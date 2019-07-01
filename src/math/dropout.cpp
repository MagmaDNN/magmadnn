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
void dropout(Tensor<T> *x, Tensor<T> *out) {
    if (out->get_memory_type() == HOST) {
        printf("Dropout for CPU not yet supported.\n");
    } else {
        fprintf(stderr, "For dropout on GPU, please use dropout_device\n");
    }
}
template void dropout(Tensor<int> *x, Tensor<int> *out);
template void dropout(Tensor<float> *x, Tensor<float> *out);
template void dropout(Tensor<double> *x, Tensor<double> *out);


template <typename T>
void dropout_grad(Tensor<T> *grad, Tensor<T> *out) {
    if (out->get_memory_type() == HOST) {
        printf("Dropout for CPU not yet supported.\n");
    }
    #if defined(_HAS_CUDA_)
    else {
        fprintf(stderr, "For dropout_grad on GPU, please use dropout_grad_device\n");
    }
    #endif
}
template void dropout_grad(Tensor<int> *grad, Tensor<int> *out);
template void dropout_grad(Tensor<float> *grad, Tensor<float> *out);
template void dropout_grad(Tensor<double> *grad, Tensor<double> *out);



#if defined(_HAS_CUDA_)
template <typename T>
void dropout_device(Tensor<T> *x, Tensor<T> *out, cudnn_dropout_settings_t settings, cudnn_dropout_shared_settings_t shared) {
    cudnnErrchk( cudnnDropoutForward(settings.handle,
        shared.dropoutDesc,
        settings.xdesc,
        (void *)x->get_ptr(),
        settings.ydesc,
        (void *)out->get_ptr(),
        shared.reserveSpace,
        shared.reserveSpaceSizeInBytes) 
    );
}
template void dropout_device(Tensor<int> *x, Tensor<int> *out, cudnn_dropout_settings_t settings, cudnn_dropout_shared_settings_t shared);
template void dropout_device(Tensor<float> *x, Tensor<float> *out, cudnn_dropout_settings_t settings, cudnn_dropout_shared_settings_t shared);
template void dropout_device(Tensor<double> *x, Tensor<double> *out, cudnn_dropout_settings_t settings, cudnn_dropout_shared_settings_t shared);

template <typename T>
void dropout_grad_device(Tensor<T> *grad, Tensor<T> *out, cudnn_dropout_grad_settings_t settings, cudnn_dropout_shared_settings_t shared) {
    cudnnErrchk( cudnnDropoutBackward(settings.handle,
        shared.dropoutDesc,
        settings.dydesc,
        (void *)grad->get_ptr(),
        settings.dxdesc,
        (void *)out->get_ptr(),
        shared.reserveSpace,
        shared.reserveSpaceSizeInBytes) 
    );
}
template void dropout_grad_device(Tensor<int> *grad, Tensor<int> *out, cudnn_dropout_grad_settings_t settings, cudnn_dropout_shared_settings_t shared);
template void dropout_grad_device(Tensor<float> *grad, Tensor<float> *out, cudnn_dropout_grad_settings_t settings, cudnn_dropout_shared_settings_t shared);
template void dropout_grad_device(Tensor<double> *grad, Tensor<double> *out, cudnn_dropout_grad_settings_t settings, cudnn_dropout_shared_settings_t shared);

#endif

}
}