/**
 * @file relu.cpp
 * @author Daniel Nichols
 * @version 0.1.0
 * @date 2019-06-21
 * 
 * @copyright Copyright (c) 2019
 */
#include "math/relu.h"

namespace magmadnn {
namespace math {

template <typename T>
void relu(Tensor<T> *x, Tensor<T> *out) {
    assert( T_IS_SAME_MEMORY_TYPE(x, out) );

    if (out->get_memory_type() == HOST) {
        T *x_ptr = x->get_ptr();
        T *out_ptr = out->get_ptr();
        unsigned int size = out->get_size();

        for (unsigned int i = 0; i < size; i++) {
            out_ptr[i] = (x_ptr[i] > static_cast<T>(0)) ? x_ptr[i] : static_cast<T>(0);
        }
    }
    #if defined(_HAS_CUDA_)
    else {
        fprintf(stderr, "For GPU relu please use relu_device\n");
    }
    #endif
}
template void relu(Tensor<int> *x, Tensor<int> *out);
template void relu(Tensor<float> *x, Tensor<float> *out);
template void relu(Tensor<double> *x, Tensor<double> *out);

template <typename T>
void relu_grad(Tensor<T> *x, Tensor<T> *relu_out, Tensor<T> *grad, Tensor<T> *out) {
    assert( T_IS_SAME_MEMORY_TYPE(x, grad) && T_IS_SAME_MEMORY_TYPE(grad, out) );

    if (out->get_memory_type() == HOST) {
        T *x_ptr = x->get_ptr();
        T *grad_ptr = grad->get_ptr();
        T *out_ptr = out->get_ptr();
        unsigned int size = out->get_size();

        for (unsigned int i = 0; i < size; i++) {
            out_ptr[i] = (x_ptr[i] > static_cast<T>(0)) ? grad_ptr[i] : static_cast<T>(0);
        }
    }
    #if defined(_HAS_CUDA_)
    else {
        fprintf(stderr, "For GPU relu_grad please use relu_grad_device\n");
    }
    #endif
}
template void relu_grad(Tensor<int> *x, Tensor<int> *relu_out, Tensor<int> *grad, Tensor<int> *out);
template void relu_grad(Tensor<float> *x, Tensor<float> *relu_out, Tensor<float> *grad, Tensor<float> *out);
template void relu_grad(Tensor<double> *x, Tensor<double> *relu_out, Tensor<double> *grad, Tensor<double> *out);

#if defined(_HAS_CUDA_)
template <typename T>
void relu_device(Tensor<T> *x, Tensor<T> *out, relu_cudnn_settings_t settings) {
    T alpha = static_cast<T>(1), beta = static_cast<T>(0);
    cudnnErrchk( cudnnActivationForward(::magmadnn::internal::MAGMADNN_SETTINGS->cudnn_handle,
        settings.descriptor,
        &alpha,
        x->get_cudnn_tensor_descriptor(),
        x->get_ptr(),
        &beta,
        out->get_cudnn_tensor_descriptor(),
        out->get_ptr())
    );
}
template void relu_device(Tensor<int> *x, Tensor<int> *out, relu_cudnn_settings_t settings);
template void relu_device(Tensor<float> *x, Tensor<float> *out, relu_cudnn_settings_t settings);
template void relu_device(Tensor<double> *x, Tensor<double> *out, relu_cudnn_settings_t settings);

template <typename T>
void relu_grad_device(Tensor<T> *x, Tensor<T> *relu_out, Tensor<T> *grad, Tensor<T> *out, relu_cudnn_settings_t settings) {
    T alpha = static_cast<T>(1), beta = static_cast<T>(0);
    cudnnErrchk( cudnnActivationBackward(::magmadnn::internal::MAGMADNN_SETTINGS->cudnn_handle,
        settings.descriptor,
        &alpha,
        relu_out->get_cudnn_tensor_descriptor(),
        relu_out->get_ptr(),
        grad->get_cudnn_tensor_descriptor(),
        grad->get_ptr(),
        x->get_cudnn_tensor_descriptor(),
        x->get_ptr(),
        &beta,
        out->get_cudnn_tensor_descriptor(),
        out->get_ptr())
    );
}
template void relu_grad_device(Tensor<int> *x, Tensor<int> *relu_out, Tensor<int> *grad, Tensor<int> *out, relu_cudnn_settings_t settings);
template void relu_grad_device(Tensor<float> *x, Tensor<float> *relu_out, Tensor<float> *grad, Tensor<float> *out, relu_cudnn_settings_t settings);
template void relu_grad_device(Tensor<double> *x, Tensor<double> *relu_out, Tensor<double> *grad, Tensor<double> *out, relu_cudnn_settings_t settings);
#endif

}
}