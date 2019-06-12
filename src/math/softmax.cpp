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
void softmax(Tensor<T> *x, Tensor<T> *out) {
    if (out->get_memory_type() == HOST) {
        assert( T_IS_MATRIX(x) && T_IS_MATRIX(out) );

        T *x_ptr = x->get_ptr();
        T *out_ptr = out->get_ptr();
        T x_max = x_ptr[0];
        T exps_sum = (T) 0;
        unsigned int x_rows = x->get_shape(0);
        unsigned int x_cols = x->get_shape(1);

        /* for each row in x, compute the softmax function */
        for (unsigned int i = 0; i < x_rows; i++) {
            x_max = x_ptr[i * x_cols + 0];
            exps_sum = (T) 0;

            /* compute max of this row */
            for (unsigned int j = 1; j < x_cols; j++) {
                if (x_ptr[i*x_cols + j] > x_max) {
                    x_max = x_ptr[i*x_cols + j];
                }
            }

            /* softmax = exp(x-max). also keep track of sum of exps */
            for (unsigned int j = 0; j < x_cols; j++) {
                out_ptr[i*x_cols + j] = exp(x_ptr[i*x_cols + j] - x_max);
                exps_sum += out_ptr[i*x_cols + j];
            }

            /* normalize by the sum */
            for (unsigned int j = 0; j < x_cols; j++) {
                out_ptr[i*x_cols + j] /= exps_sum;
            }
        }
    } else {
        fprintf(stderr, "For softmax on GPU, please use softmax_device\n");
    }
}
template void softmax(Tensor<int> *x, Tensor<int> *out);
template void softmax(Tensor<float> *x, Tensor<float> *out);
template void softmax(Tensor<double> *x, Tensor<double> *out);


template <typename T>
void softmax_grad(Tensor<T> *softmax, Tensor<T> *grad, Tensor<T> *out) {
    if (out->get_memory_type() == HOST) {
        fprintf(stderr, "softmax_grad on HOST not yet implemented\n");
    }
    #if defined(_HAS_CUDA_)
    else {
        fprintf(stderr, "For softmax_grad on GPU, please use softmax_grad_device\n");
    }
    #endif
}
template void softmax_grad(Tensor<int> *softmax, Tensor<int> *grad, Tensor<int> *out);
template void softmax_grad(Tensor<float> *softmax, Tensor<float> *grad, Tensor<float> *out);
template void softmax_grad(Tensor<double> *softmax, Tensor<double> *grad, Tensor<double> *out);



#if defined(_HAS_CUDA_)
template <typename T>
void softmax_device(Tensor<T> *x, Tensor<T> *out, cudnn_softmax_settings_t settings) {
    T alpha = (T) 1, beta = (T) 0;
    cudnnSoftmaxForward(settings.handle, settings.alg, settings.mode, (void *)&alpha, settings.xdesc, 
        (void *)x->get_ptr(), (void *)&beta, settings.ydesc, (void *)out->get_ptr());
}
template void softmax_device(Tensor<int> *x, Tensor<int> *out, cudnn_softmax_settings_t settings);
template void softmax_device(Tensor<float> *x, Tensor<float> *out, cudnn_softmax_settings_t settings);
template void softmax_device(Tensor<double> *x, Tensor<double> *out, cudnn_softmax_settings_t settings);

template <typename T>
void softmax_grad_device(Tensor<T> *y, Tensor<T> *grad, Tensor<T> *out, cudnn_softmax_grad_settings_t settings) {
    T alpha = (T) 1, beta = (T) 0;
    cudnnSoftmaxBackward(settings.handle, settings.alg, settings.mode, (void *)&alpha, settings.ydesc, (void *)y->get_ptr(),
        settings.dydesc, (void *)grad->get_ptr(), (void *)&beta, settings.dxdesc, (void *)out->get_ptr());
}
template void softmax_grad_device(Tensor<int> *y, Tensor<int> *grad, Tensor<int> *out, cudnn_softmax_grad_settings_t settings);
template void softmax_grad_device(Tensor<float> *y, Tensor<float> *grad, Tensor<float> *out, cudnn_softmax_grad_settings_t settings);
template void softmax_grad_device(Tensor<double> *y, Tensor<double> *grad, Tensor<double> *out, cudnn_softmax_grad_settings_t settings);

#endif

}
}