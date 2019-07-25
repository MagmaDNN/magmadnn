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

template <>
void conv2d<CPU>(const Tensor &x, const Tensor &w, Tensor &out) {
    MAGMADNN_ASSERT(::magmadnn::utilities::do_tensors_match(out.dtype(), out.get_memory_type(), {x, w, out}),
                    "invalid input tensors");
    if (out.get_memory_type() == HOST) {
        LOG(ERROR) << "__Conv2d CPU not supported yet.\n";
    }
}

template <>
void conv2d_grad_data<CPU>(const Tensor &w, const Tensor &grad, Tensor &out) {
    MAGMADNN_ASSERT(::magmadnn::utilities::do_tensors_match(out.dtype(), out.get_memory_type(), {w, grad, out}),
                    "invalid input tensors");

    if (out.get_memory_type() == HOST) {
        LOG(ERROR) << "__Conv2d_grad_data CPU not supported yet.\n";
    }
}

template <>
void conv2d_grad_filter<CPU>(const Tensor &x, const Tensor &grad, Tensor &out) {
    MAGMADNN_ASSERT(::magmadnn::utilities::do_tensors_match(out.dtype(), out.get_memory_type(), {x, grad, out}),
                    "invalid input tensors");

    if (out.get_memory_type() == HOST) {
        LOG(ERROR) << "__Conv2d_grad_filter CPU not supported yet.\n";
    }
}

#if defined(_HAS_CUDA_)

template <>
void conv2d<GPU>(const Tensor &x, const Tensor &w, Tensor &out, conv2d_cudnn_settings settings) {
    FOR_ALL_DTYPES(out.dtype(), T, {
        T alpha = static_cast<T>(1), beta = static_cast<T>(0);
        cudnnErrchk(cudnnConvolutionForward(
            ::magmadnn::internal::MAGMADNN_SETTINGS->cudnn_handle, &alpha, x.get_cudnn_tensor_descriptor(),
            x.get_ptr<T>(), settings.filter_desc, w.get_ptr<T>(), settings.conv_desc, settings.algo, settings.workspace,
            settings.workspace_size, &beta, out.get_cudnn_tensor_descriptor(), out.get_ptr<T>()));
    })
}

template <>
void conv2d_grad_data<GPU>(const Tensor &w, const Tensor &grad, Tensor &out, conv2d_cudnn_settings settings) {
    FOR_ALL_DTYPES(out.dtype(), T, {
        T alpha = static_cast<T>(1), beta = static_cast<T>(0);
        cudnnErrchk(cudnnConvolutionBackwardData(
            ::magmadnn::internal::MAGMADNN_SETTINGS->cudnn_handle, &alpha, settings.filter_desc, w.get_ptr<T>(),
            grad.get_cudnn_tensor_descriptor(), grad.get_ptr<T>(), settings.conv_desc, settings.bwd_data_algo,
            settings.grad_data_workspace, settings.grad_data_workspace_size, &beta, out.get_cudnn_tensor_descriptor(),
            out.get_ptr<T>()));
    })
}

template <>
void conv2d_grad_filter<GPU>(const Tensor &x, const Tensor &grad, Tensor &out, conv2d_cudnn_settings settings) {
    FOR_ALL_DTYPES(out.dtype(), T, {
        T alpha = static_cast<T>(1), beta = static_cast<T>(0);
        cudnnErrchk(cudnnConvolutionBackwardFilter(
            ::magmadnn::internal::MAGMADNN_SETTINGS->cudnn_handle, &alpha, x.get_cudnn_tensor_descriptor(),
            x.get_ptr<T>(), grad.get_cudnn_tensor_descriptor(), grad.get_ptr<T>(), settings.conv_desc,
            settings.bwd_filter_algo, settings.grad_filter_workspace, settings.grad_filter_workspace_size, &beta,
            settings.filter_desc, out.get_ptr<T>()));
    })
}
#endif

}  // namespace math
}  // namespace magmadnn