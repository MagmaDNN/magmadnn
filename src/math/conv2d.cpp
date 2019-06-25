/**
 * @file conv2d.cpp
 * @author Daniel Nichols
 * @version 0.1.0
 * @date 2019-06-24
 * 
 * @copyright Copyright (c) 2019
 */
#include "math/conv2d.h"

namespace magmadnn {
namespace math {

template <typename T>
void conv2d(Tensor<T> *x, Tensor<T> *w, Tensor<T> *out) {
    assert( T_IS_SAME_MEMORY_TYPE(x, w) && T_IS_SAME_MEMORY_TYPE(w, out) );
    
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


#if defined(_HAS_CUDA_)

template <typename T>
void conv2d_device(Tensor<T> *x, Tensor<T> *w, Tensor<T> *out, conv2d_cudnn_settings settings) {
    T alpha = static_cast<T>(1), beta = static_cast<T>(0);
    cudnnErrchk( cudnnConvolutionForward(
        ::magmadnn::internal::MAGMADNN_SETTINGS->cudnn_handle,
        &alpha,
        x->get_cudnn_tensor_descriptor(),
        x->get_ptr(),
        settings.filter_desc,
        w->get_cudnn_tensor_descriptor(),
        settings.conv_desc,
        settings.algo,
        settings.workspace,
        settings.workspace_size,
        &beta,
        out->get_cudnn_tensor_descriptor(),
        out->get_ptr()
        )
    );
}
template void conv2d_device(Tensor<int> *x, Tensor<int> *w, Tensor<int> *out, conv2d_cudnn_settings settings);
template void conv2d_device(Tensor<float> *x, Tensor<float> *w, Tensor<float> *out, conv2d_cudnn_settings settings);
template void conv2d_device(Tensor<double> *x, Tensor<double> *w, Tensor<double> *out, conv2d_cudnn_settings settings);

#endif

}
}