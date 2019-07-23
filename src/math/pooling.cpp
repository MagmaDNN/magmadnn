/**
 * @file pooling.cpp
 * @author Sedrick Keh
 * @version 1.0
 * @date 2019-07-08
 *
 * @copyright Copyright (c) 2019
 */
#include "math/pooling.h"

namespace magmadnn {
namespace math {

template <typename T>
void pooling(const Tensor &x, Tensor &out) {
    // assert(T_IS_SAME_MEMORY_TYPE(x, out));

    if (out.get_memory_type() == HOST) {
        LOG(ERROR) << "__pooling CPU not supported yet.\n";
    }
#if defined(_HAS_CUDA_)
    else {
        LOG(ERROR) << "For Pooling GPU please use pooling_device.\n";
    }
#endif
}
#define COMPILE_POOLING(type) template void pooling<type>(const Tensor &, Tensor &);
CALL_FOR_ALL_TYPES(COMPILE_POOLING)
#undef COMPILE_POOLING

template <typename T>
void pooling_grad(const Tensor &x, const Tensor &y, const Tensor &grad, Tensor &out) {
    // assert(T_IS_SAME_MEMORY_TYPE(x, y) && T_IS_SAME_MEMORY_TYPE(y, grad) && T_IS_SAME_MEMORY_TYPE(grad, out));

    if (out.get_memory_type() == HOST) {
        LOG(ERROR) << "__Pooling_grad CPU not supported yet.\n";
    }
#if defined(_HAS_CUDA_)
    else {
        LOG(ERROR) << "For pooling_grad GPU please use pooling_grad_device.\n";
    }
#endif
}
#define COMPILE_POOLINGGRAD(type) \
    template void pooling_grad<type>(const Tensor &, const Tensor &, const Tensor &, Tensor &);
CALL_FOR_ALL_TYPES(COMPILE_POOLINGGRAD)
#undef COMPILE_POOLINGGRAD

#if defined(_HAS_CUDA_)

template <typename T>
void pooling_device(const Tensor &x, Tensor &out, cudnn_pooling_settings_t settings) {
    T alpha = static_cast<T>(1), beta = static_cast<T>(0);
    cudnnErrchk(cudnnPoolingForward(::magmadnn::internal::MAGMADNN_SETTINGS->cudnn_handle, settings.poolingDesc, &alpha,
                                    x.get_cudnn_tensor_descriptor(), x.get_ptr<T>(), &beta,
                                    out.get_cudnn_tensor_descriptor(), out.get_ptr<T>()));
}
#define COMPILE_POOLING_DEVICE(type) \
    template void pooling_device<type>(const Tensor &, Tensor &, cudnn_pooling_settings_t);
CALL_FOR_ALL_TYPES(COMPILE_POOLING_DEVICE)
#undef COMPILE_POOLING_DEVICE

template <typename T>
void pooling_grad_device(const Tensor &x, const Tensor &y, const Tensor &grad, Tensor &out,
                         cudnn_pooling_settings_t settings) {
    T alpha = static_cast<T>(1), beta = static_cast<T>(0);
    cudnnErrchk(cudnnPoolingBackward(
        ::magmadnn::internal::MAGMADNN_SETTINGS->cudnn_handle, settings.poolingDesc, &alpha,
        y.get_cudnn_tensor_descriptor(), y.get_ptr<T>(), grad.get_cudnn_tensor_descriptor(), grad.get_ptr<T>(),
        x.get_cudnn_tensor_descriptor(), x.get_ptr<T>(), &beta, out.get_cudnn_tensor_descriptor(), out.get_ptr<T>()));
}
#define COMPILE_POOLINGGRAD_DEVICE(type)                                                              \
    template void pooling_grad_device<type>(const Tensor &, const Tensor &, const Tensor &, Tensor &, \
                                            cudnn_pooling_settings_t);
CALL_FOR_ALL_TYPES(COMPILE_POOLINGGRAD_DEVICE)
#undef COMPILE_POOLINGGRAD_DEVICE

#endif

}  // namespace math
}  // namespace magmadnn