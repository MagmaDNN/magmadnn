/**
 * @file conv2d.cpp
 * @author Daniel Nichols
 * @author Florent Lopez
 * @author Rocco Febbo
 * @version 1.0
 * @date 2019-06-24
 *
 * @copyright Copyright (c) 2019
 */
#include "math/conv2d.h"

#include <cassert>

namespace magmadnn {
namespace math {

template <typename T>
void conv2d(Tensor<T> *x, Tensor<T> *w, Tensor<T> *out, const int pad_h, const int pad_w, const int vertical_stride,
            const int horizontal_stride, const int dilation_h, const int dilation_w) {
    assert(T_IS_SAME_MEMORY_TYPE(x, w) && T_IS_SAME_MEMORY_TYPE(w, out));

    if (out->get_memory_type() == HOST) {
        if (out == NULL) {
            fprintf(stderr, "Error: Conv2D::unallocated output.\n");
            return;
        }

        std::vector<unsigned int> out_shape = out->get_shape();
        std::vector<unsigned int> w_shape = w->get_shape();
        // fprintf(stderr, "__Conv2d CPU not supported yet.\n");

        unsigned int o_n, o_c, o_h, o_w, k_n, k_c, k_h, k_w;  // iteration variables
        unsigned int No, Co, Ho, Wo, Ck, Hk, Wk, Ci;          // shorthand for tensor dims
        if (out_shape.size() == 4) {
            No = out_shape[0];
            Co = out_shape[1];
            Ho = out_shape[2];
            Wo = out_shape[3];
        } else if (out_shape.size() == 3) {
            No = 1;
            Co = out_shape[0];
            Ho = out_shape[1];
            Wo = out_shape[2];
        } else {
            fprintf(stderr, "Error: Conv2D::invalid output shape size.\n");
            return;
        }

        if (w_shape.size() == 4) {
            Ck = w_shape[1];
            Hk = w_shape[2];
            Wk = w_shape[3];
        } else {
            fprintf(stderr, "Error: Conv2D::invalid filter shape size.\n");
            return;
        }

        // crop
        for (o_n = 0; o_n < No; o_n++) {
            for (o_c = 0; o_c < Co; o_c++) {
                for (o_h = 0; o_h < Ho; o_h++) {
                    for (o_w = 0; o_w < Wo; o_w++) {
                        float val = 0;

                        // traverse kernel and sum input image values
                        for (k_c = 0; k_c < Ck; k_c++) {
                            for (k_h = 0; k_h < Hk; k_h++) {
                                for (k_w = 0; k_w < Wk; k_w++) {
                                    unsigned int in_h = o_h * vertical_stride - pad_h + k_h * dilation_h;
                                    unsigned int in_w = o_w * horizontal_stride - pad_w + k_w * dilation_w;

                                    if (in_h < out_shape[1] && in_w < out_shape[2]) {
                                        val += w->get({o_c, k_c, k_h, k_w}) * x->get({o_n, k_c, in_h, in_w});
                                    }
                                }
                            }
                        }

                        out->set({o_n, o_c, o_h, o_w}, val);
                    }
                }
            }
        }

    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {
        fprintf(stderr, "For Conv2d GPU please use conv2d_device.\n");
    }
#endif
}
template void conv2d(Tensor<int> *x, Tensor<int> *w, Tensor<int> *out, const int pad_h, const int pad_w,
                     const int vertical_stride, const int horizontal_stride, const int dilation_h,
                     const int dilation_w);
template void conv2d(Tensor<float> *x, Tensor<float> *w, Tensor<float> *out, const int pad_h, const int pad_w,
                     const int vertical_stride, const int horizontal_stride, const int dilation_h,
                     const int dilation_w);
template void conv2d(Tensor<double> *x, Tensor<double> *w, Tensor<double> *out, const int pad_h, const int pad_w,
                     const int vertical_stride, const int horizontal_stride, const int dilation_h,
                     const int dilation_w);

template <typename T>
void conv2d_grad_data(Tensor<T> *w, Tensor<T> *grad, Tensor<T> *out) {
    assert(T_IS_SAME_MEMORY_TYPE(w, grad) && T_IS_SAME_MEMORY_TYPE(grad, out));

    if (out->get_memory_type() == HOST) {
        fprintf(stderr, "__Conv2d_grad_data CPU not supported yet.\n");
    }
#if defined(MAGMADNN_HAVE_CUDA)
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
#if defined(MAGMADNN_HAVE_CUDA)
    else {
        fprintf(stderr, "For Conv2d_grad_filter GPU please use conv2d_grad_filter_device.\n");
    }
#endif
}
template void conv2d_grad_filter(Tensor<int> *x, Tensor<int> *grad, Tensor<int> *out);
template void conv2d_grad_filter(Tensor<float> *x, Tensor<float> *grad, Tensor<float> *out);
template void conv2d_grad_filter(Tensor<double> *x, Tensor<double> *grad, Tensor<double> *out);

#if defined(MAGMADNN_HAVE_CUDA)

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