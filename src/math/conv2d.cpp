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
void conv2d(const Tensor &x, const Tensor &w, Tensor &out) {
    // assert(T_IS_SAME_MEMORY_TYPE(x, w) && T_IS_SAME_MEMORY_TYPE(w, out));

    MAGMADNN_ASSERT(TYPES_MATCH(T, x.dtype()) && TYPES_MATCH(T, w.dtype()) && TYPES_MATCH(T, out.dtype()),
                    "invalid tensor types");

    if (out.get_memory_type() == HOST) {
        LOG(ERROR) << "__Conv2d CPU not supported yet.\n";
    }
#if defined(_HAS_CUDA_)
    else {
        LOG(ERROR) << "For Conv2d GPU please use conv2d_device.\n";
    }
#endif
}
#define COMPILE_CONV2D(type) template void conv2d<type>(const Tensor &, const Tensor &, Tensor &);
CALL_FOR_ALL_TYPES(COMPILE_CONV2D)
#undef COMPILE_CONV2D

template <typename T>
void conv2d_grad_data(const Tensor &w, const Tensor &grad, Tensor &out) {
    // assert(T_IS_SAME_MEMORY_TYPE(w, grad) && T_IS_SAME_MEMORY_TYPE(grad, out));

    MAGMADNN_ASSERT(TYPES_MATCH(T, w.dtype()) && TYPES_MATCH(T, grad.dtype()) && TYPES_MATCH(T, out.dtype()),
                    "invalid tensor types");

    if (out.get_memory_type() == HOST) {
        LOG(ERROR) << "__Conv2d_grad_data CPU not supported yet.\n";
    }
#if defined(_HAS_CUDA_)
    else {
        LOG(ERROR) << "For Conv2d_grad_data GPU please use conv2d_grad_data_device.\n";
    }
#endif
}
#define comp(type) template void conv2d_grad_data<type>(const Tensor &, const Tensor &, Tensor &);
CALL_FOR_ALL_TYPES(comp)
#undef comp

template <typename T>
void conv2d_grad_filter(const Tensor &x, const Tensor &grad, Tensor &out) {
    // assert(T_IS_SAME_MEMORY_TYPE(w, grad) && T_IS_SAME_MEMORY_TYPE(grad, out));

    MAGMADNN_ASSERT(TYPES_MATCH(T, x.dtype()) && TYPES_MATCH(T, grad.dtype()) && TYPES_MATCH(T, out.dtype()),
                    "invalid tensor types");

    if (out.get_memory_type() == HOST) {
        LOG(ERROR) << "__Conv2d_grad_filter CPU not supported yet.\n";
    }
#if defined(_HAS_CUDA_)
    else {
        LOG(ERROR) << "For Conv2d_grad_filter GPU please use conv2d_grad_filter_device.\n";
    }
#endif
}
#define comp(type) template void conv2d_grad_filter<type>(const Tensor &, const Tensor &, Tensor &);
CALL_FOR_ALL_TYPES(comp)
#undef comp

#if defined(_HAS_CUDA_)

template <typename T>
void conv2d_device(const Tensor &x, const Tensor &w, Tensor &out, conv2d_cudnn_settings settings) {
    T alpha = static_cast<T>(1), beta = static_cast<T>(0);
    cudnnErrchk(cudnnConvolutionForward(
        ::magmadnn::internal::MAGMADNN_SETTINGS->cudnn_handle, &alpha, x.get_cudnn_tensor_descriptor(), x.get_ptr<T>(),
        settings.filter_desc, w.get_ptr<T>(), settings.conv_desc, settings.algo, settings.workspace,
        settings.workspace_size, &beta, out.get_cudnn_tensor_descriptor(), out.get_ptr<T>()));
}
#define comp(type) template void conv2d_device<type>(const Tensor &, const Tensor &, Tensor &, conv2d_cudnn_settings);
CALL_FOR_ALL_TYPES(comp)
#undef comp

template <typename T>
void conv2d_grad_data_device(const Tensor &w, const Tensor &grad, Tensor &out, conv2d_cudnn_settings settings) {
    T alpha = static_cast<T>(1), beta = static_cast<T>(0);
    cudnnErrchk(cudnnConvolutionBackwardData(::magmadnn::internal::MAGMADNN_SETTINGS->cudnn_handle, &alpha,
                                             settings.filter_desc, w.get_ptr<T>(), grad.get_cudnn_tensor_descriptor(),
                                             grad.get_ptr<T>(), settings.conv_desc, settings.bwd_data_algo,
                                             settings.grad_data_workspace, settings.grad_data_workspace_size, &beta,
                                             out.get_cudnn_tensor_descriptor(), out.get_ptr<T>()));
}
#define comp(type) \
    template void conv2d_grad_data_device<type>(const Tensor &, const Tensor &, Tensor &, conv2d_cudnn_settings);
CALL_FOR_ALL_TYPES(comp)
#undef comp

template <typename T>
void conv2d_grad_filter_device(const Tensor &x, const Tensor &grad, Tensor &out, conv2d_cudnn_settings settings) {
    T alpha = static_cast<T>(1), beta = static_cast<T>(0);
    cudnnErrchk(cudnnConvolutionBackwardFilter(
        ::magmadnn::internal::MAGMADNN_SETTINGS->cudnn_handle, &alpha, x.get_cudnn_tensor_descriptor(), x.get_ptr<T>(),
        grad.get_cudnn_tensor_descriptor(), grad.get_ptr<T>(), settings.conv_desc, settings.bwd_filter_algo,
        settings.grad_filter_workspace, settings.grad_filter_workspace_size, &beta, settings.filter_desc,
        out.get_ptr<T>()));
}
#define comp(type) \
    template void conv2d_grad_filter_device<type>(const Tensor &, const Tensor &, Tensor &, conv2d_cudnn_settings);
CALL_FOR_ALL_TYPES(comp)
#undef comp

#endif

}  // namespace math
}  // namespace magmadnn