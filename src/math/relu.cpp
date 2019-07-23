/**
 * @file relu.cpp
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-06-21
 *
 * @copyright Copyright (c) 2019
 */
#include "math/relu.h"

namespace magmadnn {
namespace math {

template <typename T>
void relu(const Tensor &x, Tensor &out) {
    // assert(T_IS_SAME_MEMORY_TYPE(x, out));
    MAGMADNN_ASSERT(TYPES_MATCH(T, x.dtype()) && TYPES_MATCH(T, x.dtype()), "invalid tensor type")

    if (out.get_memory_type() == HOST) {
        const T *x_ptr = x.get_ptr<T>();
        T *out_ptr = out.get_ptr<T>();
        size_t size = out.size();

        for (size_t i = 0; i < size; i++) {
            out_ptr[i] = (x_ptr[i] > static_cast<T>(0)) ? x_ptr[i] : static_cast<T>(0);
        }
    }
#if defined(_HAS_CUDA_)
    else {
        fprintf(stderr, "For GPU relu please use relu_device\n");
    }
#endif
}
#define COMPILE_RELU(type) template void relu<type>(const Tensor &, Tensor &);
CALL_FOR_ALL_TYPES(COMPILE_RELU)
#undef COMPILE_RELU

template <typename T>
void relu_grad(const Tensor &x, const Tensor &relu_out, const Tensor &grad, Tensor &out) {
    // assert(T_IS_SAME_MEMORY_TYPE(x, grad) && T_IS_SAME_MEMORY_TYPE(grad, out));
    assert(TYPES_MATCH(T, x.dtype()) && TYPES_MATCH(T, relu_out.dtype()) && TYPES_MATCH(T, grad.dtype()) &&
           TYPES_MATCH(T, out.dtype()));

    if (out.get_memory_type() == HOST) {
        const T *x_ptr = x.get_ptr<T>();
        const T *grad_ptr = grad.get_ptr<T>();
        T *out_ptr = out.get_ptr<T>();
        size_t size = out.size();

        for (size_t i = 0; i < size; i++) {
            out_ptr[i] = (x_ptr[i] > static_cast<T>(0)) ? grad_ptr[i] : static_cast<T>(0);
        }
    }
#if defined(_HAS_CUDA_)
    else {
        fprintf(stderr, "For GPU relu_grad please use relu_grad_device\n");
    }
#endif
}
#define COMPILE_RELUGRAD(type) template void relu_grad<type>(const Tensor &, const Tensor &, const Tensor &, Tensor &);
CALL_FOR_ALL_TYPES(COMPILE_RELUGRAD)
#undef COMPILE_RELUGRAD

#if defined(_HAS_CUDA_)
template <typename T>
void relu_device(const Tensor &x, Tensor &out, relu_cudnn_settings_t settings) {
    T alpha = static_cast<T>(1), beta = static_cast<T>(0);
    cudnnErrchk(cudnnActivationForward(::magmadnn::internal::MAGMADNN_SETTINGS->cudnn_handle, settings.descriptor,
                                       &alpha, x.get_cudnn_tensor_descriptor(), x.get_ptr<T>(), &beta,
                                       out.get_cudnn_tensor_descriptor(), out.get_ptr<T>()));
}
#define COMPILE_RELU_DEVICE(type) template void relu_device<type>(const Tensor &, Tensor &, relu_cudnn_settings_t);
CALL_FOR_ALL_TYPES(COMPILE_RELU_DEVICE)
#undef COMPILE_RELU_DEVICE

template <typename T>
void relu_grad_device(const Tensor &x, const Tensor &relu_out, const Tensor &grad, Tensor &out,
                      relu_cudnn_settings_t settings) {
    T alpha = static_cast<T>(1), beta = static_cast<T>(0);
    cudnnErrchk(cudnnActivationBackward(::magmadnn::internal::MAGMADNN_SETTINGS->cudnn_handle, settings.descriptor,
                                        &alpha, relu_out.get_cudnn_tensor_descriptor(), relu_out.get_ptr<T>(),
                                        grad.get_cudnn_tensor_descriptor(), grad.get_ptr<T>(),
                                        x.get_cudnn_tensor_descriptor(), x.get_ptr<T>(), &beta,
                                        out.get_cudnn_tensor_descriptor(), out.get_ptr<T>()));
}
#define COMPILE_RELUGRAD_DEVICE(type)                                                              \
    template void relu_grad_device<type>(const Tensor &, const Tensor &, const Tensor &, Tensor &, \
                                         relu_cudnn_settings_t);
CALL_FOR_ALL_TYPES(COMPILE_RELUGRAD_DEVICE)
#undef COMPILE_RELUGRAD_DEVICE

#endif

}  // namespace math
}  // namespace magmadnn