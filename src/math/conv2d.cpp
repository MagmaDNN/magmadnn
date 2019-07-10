/**
 * @file conv2d.cpp
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-06-24
 *
 * @copyright Copyright (c) 2019
 */
#include "math/conv2d.h"

namespace magmadnn {
namespace math {

template <typename T>
void conv2d(Tensor<T> *x, Tensor<T> *w, Tensor<T> *out) {
    assert(T_IS_SAME_MEMORY_TYPE(x, w) && T_IS_SAME_MEMORY_TYPE(w, out));

    if (out->get_memory_type() == HOST) {
        fprintf(stderr, "__Conv2d CPU not supported yet.\n");
    }
#if defined(_HAS_CUDA_)
    else {
        fprintf(stderr, "For Conv2d GPU please use conv2d_device.\n");
    }
#endif
}
template void conv2d(Tensor<int> *x, Tensor<int> *w, Tensor<int> *out);
template void conv2d(Tensor<float> *x, Tensor<float> *w, Tensor<float> *out);
template void conv2d(Tensor<double> *x, Tensor<double> *w, Tensor<double> *out);

template <typename T>
void conv2d_grad_data(Tensor<T> *w, Tensor<T> *grad, Tensor<T> *out) {
    assert(T_IS_SAME_MEMORY_TYPE(w, grad) && T_IS_SAME_MEMORY_TYPE(grad, out));

    if (out->get_memory_type() == HOST) {
        fprintf(stderr, "__Conv2d_grad_data CPU not supported yet.\n");
    }
#if defined(_HAS_CUDA_)
    else {
        fprintf(stderr, "For Conv2d_grad_data GPU please use conv2d_grad_data_device.\n");
    }
#endif
}
template void conv2d_grad_data(Tensor<int> *w, Tensor<int> *grad, Tensor<int> *out);
template void conv2d_grad_data(Tensor<float> *w, Tensor<float> *grad, Tensor<float> *out);
template void conv2d_grad_data(Tensor<double> *w, Tensor<double> *grad, Tensor<double> *out);

template <typename T>
void conv2d_grad_filter(Tensor<T> *w, Tensor<T> *grad, Tensor<T> *out) {
    assert(T_IS_SAME_MEMORY_TYPE(w, grad) && T_IS_SAME_MEMORY_TYPE(grad, out));

    if (out->get_memory_type() == HOST) {
        fprintf(stderr, "__Conv2d_grad_filter CPU not supported yet.\n");
    }
#if defined(_HAS_CUDA_)
    else {
        fprintf(stderr, "For Conv2d_grad_filter GPU please use conv2d_grad_filter_device.\n");
    }
#endif
}
template void conv2d_grad_filter(Tensor<int> *x, Tensor<int> *grad, Tensor<int> *out);
template void conv2d_grad_filter(Tensor<float> *x, Tensor<float> *grad, Tensor<float> *out);
template void conv2d_grad_filter(Tensor<double> *x, Tensor<double> *grad, Tensor<double> *out);

#if defined(_HAS_CUDA_)

template <typename T>
void conv2d_device(Tensor<T> *x, Tensor<T> *w, Tensor<T> *out, conv2d_cudnn_settings settings) {
    T alpha = static_cast<T>(1), beta = static_cast<T>(0);
    cudnnErrchk(cudnnConvolutionForward(
        ::magmadnn::internal::MAGMADNN_SETTINGS->cudnn_handle, &alpha, x->get_cudnn_tensor_descriptor(), x->get_ptr(),
        settings.filter_desc, w->get_ptr(), settings.conv_desc, settings.algo, settings.workspace,
        settings.workspace_size, &beta, out->get_cudnn_tensor_descriptor(), out->get_ptr()));
}
template void conv2d_device(Tensor<int> *x, Tensor<int> *w, Tensor<int> *out, conv2d_cudnn_settings settings);
template void conv2d_device(Tensor<float> *x, Tensor<float> *w, Tensor<float> *out, conv2d_cudnn_settings settings);
template void conv2d_device(Tensor<double> *x, Tensor<double> *w, Tensor<double> *out, conv2d_cudnn_settings settings);

template <typename T>
void conv2d_grad_data_device(Tensor<T> *w, Tensor<T> *grad, Tensor<T> *out, conv2d_cudnn_settings settings) {
    T alpha = static_cast<T>(1), beta = static_cast<T>(0);
    cudnnErrchk(cudnnConvolutionBackwardData(::magmadnn::internal::MAGMADNN_SETTINGS->cudnn_handle, &alpha,
                                             settings.filter_desc, w->get_ptr(), grad->get_cudnn_tensor_descriptor(),
                                             grad->get_ptr(), settings.conv_desc, settings.bwd_data_algo,
                                             settings.grad_data_workspace, settings.grad_data_workspace_size, &beta,
                                             out->get_cudnn_tensor_descriptor(), out->get_ptr()));
}
template void conv2d_grad_data_device(Tensor<int> *w, Tensor<int> *grad, Tensor<int> *out,
                                      conv2d_cudnn_settings settings);
template void conv2d_grad_data_device(Tensor<float> *w, Tensor<float> *grad, Tensor<float> *out,
                                      conv2d_cudnn_settings settings);
template void conv2d_grad_data_device(Tensor<double> *w, Tensor<double> *grad, Tensor<double> *out,
                                      conv2d_cudnn_settings settings);

template <typename T>
void conv2d_grad_filter_device(Tensor<T> *x, Tensor<T> *grad, Tensor<T> *out, conv2d_cudnn_settings settings) {
    T alpha = static_cast<T>(1), beta = static_cast<T>(0);
    cudnnErrchk(cudnnConvolutionBackwardFilter(
        ::magmadnn::internal::MAGMADNN_SETTINGS->cudnn_handle, &alpha, x->get_cudnn_tensor_descriptor(), x->get_ptr(),
        grad->get_cudnn_tensor_descriptor(), grad->get_ptr(), settings.conv_desc, settings.bwd_filter_algo,
        settings.grad_filter_workspace, settings.grad_filter_workspace_size, &beta, settings.filter_desc,
        out->get_ptr()));
}
template void conv2d_grad_filter_device(Tensor<int> *x, Tensor<int> *grad, Tensor<int> *out,
                                        conv2d_cudnn_settings settings);
template void conv2d_grad_filter_device(Tensor<float> *x, Tensor<float> *grad, Tensor<float> *out,
                                        conv2d_cudnn_settings settings);
template void conv2d_grad_filter_device(Tensor<double> *x, Tensor<double> *grad, Tensor<double> *out,
                                        conv2d_cudnn_settings settings);

#endif

}  // namespace math
}  // namespace magmadnn