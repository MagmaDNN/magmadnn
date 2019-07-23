/**
 * @file softmax.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-06-12
 *
 * @copyright Copyright (c) 2019
 */
#include "math/softmax.h"

namespace magmadnn {
namespace math {

template <typename T>
void softmax(const Tensor &x, Tensor &out) {
    if (out.get_memory_type() == HOST) {
        // assert(T_IS_MATRIX(x) && T_IS_MATRIX(out));

        const T *x_ptr = x.get_ptr<T>();
        T *out_ptr = out.get_ptr<T>();
        T x_max = x_ptr[0];
        T exps_sum = static_cast<T>(0);
        index_t x_rows = x.shape(0);
        index_t x_cols = x.shape(1);

        /* for each row in x, compute the softmax function */
        for (index_t i = 0; i < x_rows; i++) {
            x_max = x_ptr[i * x_cols + 0];
            exps_sum = static_cast<T>(0);

            /* compute max of this row */
            for (index_t j = 1; j < x_cols; j++) {
                if (x_ptr[i * x_cols + j] > x_max) {
                    x_max = x_ptr[i * x_cols + j];
                }
            }

            /* softmax = exp(x-max). also keep track of sum of exps */
            for (index_t j = 0; j < x_cols; j++) {
                out_ptr[i * x_cols + j] = exp(x_ptr[i * x_cols + j] - x_max);
                exps_sum += out_ptr[i * x_cols + j];
            }

            /* normalize by the sum */
            for (index_t j = 0; j < x_cols; j++) {
                out_ptr[i * x_cols + j] /= exps_sum;
            }
        }
    } else {
        LOG(ERROR) << "For softmax on GPU, please use softmax_device\n";
    }
}
#define COMPILE_SOFTMAX(type) template void softmax<type>(const Tensor &, Tensor &);
CALL_FOR_ALL_TYPES(COMPILE_SOFTMAX)
#undef COMPILE_SOFTMAX

template <typename T>
void softmax_grad(const Tensor &softmax, const Tensor &grad, Tensor &out) {
    if (out.get_memory_type() == HOST) {
        // assert(T_IS_MATRIX(softmax) && T_IS_MATRIX(out));

        /* softmax grad is: (grad - RowReduce(grad * softmax)) * softmax */
        const T *softmax_ptr = softmax.get_ptr<T>();
        const T *grad_ptr = grad.get_ptr<T>();
        T *out_ptr = out.get_ptr<T>();
        index_t n_rows = out.shape(0);
        index_t n_cols = out.shape(1);
        bool grad_is_scalar = (grad.size() == 1);
        T sum;

        for (index_t i = 0; i < n_rows; i++) {
            sum = static_cast<T>(0);
            for (index_t j = 0; j < n_cols; j++) {
                sum += grad_ptr[(grad_is_scalar) ? 0 : (i * n_cols + j)] * softmax_ptr[i * n_cols + j];
            }

            for (index_t j = 0; j < n_cols; j++) {
                out_ptr[i * n_cols + j] =
                    (grad_ptr[(grad_is_scalar) ? 0 : (i * n_cols + j)] - sum) * softmax_ptr[i * n_cols + j];
            }
        }

    }
#if defined(_HAS_CUDA_)
    else {
        LOG(ERROR) << "For softmax_grad on GPU, please use softmax_grad_device\n";
    }
#endif
}
#define COMPILE_SOFTMAXGRAD(type) template void softmax_grad<type>(const Tensor &, const Tensor &, Tensor &);
CALL_FOR_ALL_TYPES(COMPILE_SOFTMAXGRAD)
#undef COMPILE_SOFTMAXGRAD

#if defined(_HAS_CUDA_)
template <typename T>
void softmax_device(const Tensor &x, Tensor &out, cudnn_softmax_settings_t settings) {
    T alpha = static_cast<T>(1), beta = static_cast<T>(0);
    cudnnErrchk(cudnnSoftmaxForward(settings.handle, settings.alg, settings.mode, (void *) &alpha, settings.xdesc,
                                    (void *) x.get_ptr<T>(), (void *) &beta, settings.ydesc,
                                    (void *) out.get_ptr<T>()));
}
#define COMPILE_SOFTMAX_DEVICE(type) \
    template void softmax_device<type>(const Tensor &, Tensor &, cudnn_softmax_settings_t);
CALL_FOR_ALL_TYPES(COMPILE_SOFTMAX_DEVICE)
#undef COMPILE_SOFTMAX_DEVICE

template <typename T>
void softmax_grad_device(const Tensor &y, const Tensor &grad, Tensor &out, cudnn_softmax_grad_settings_t settings) {
    T alpha = static_cast<T>(1), beta = static_cast<T>(0);
    cudnnErrchk(cudnnSoftmaxBackward(settings.handle, settings.alg, settings.mode, (void *) &alpha, settings.ydesc,
                                     (void *) y.get_ptr<T>(), settings.dydesc, (void *) grad.get_ptr<T>(),
                                     (void *) &beta, settings.dxdesc, (void *) out.get_ptr<T>()));
}
#define COMPILE_SOFTMAXGRAD_DEVICE(type) \
    template void softmax_grad_device<type>(const Tensor &, const Tensor &, Tensor &, cudnn_softmax_grad_settings_t);
CALL_FOR_ALL_TYPES(COMPILE_SOFTMAXGRAD_DEVICE)
#undef COMPILE_SOFTMAXGRAD_DEVICE

#endif

}  // namespace math
}  // namespace magmadnn