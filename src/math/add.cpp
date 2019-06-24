/**
 * @file add.cpp
 * @author Daniel Nichols
 * @version 0.1.0
 * @date 2019-06-24
 * 
 * @copyright Copyright (c) 2019
 */
#include "math/add.h"

namespace magmadnn {
namespace math {

template <typename T>
void add_in_place(T alpha, Tensor<T> *x, T beta, Tensor<T> *out) {
    assert( x->get_size() == out->get_size() );
    assert( T_IS_SAME_MEMORY_TYPE(x, out) );

    if (out->get_memory_type() == HOST) {
        T *x_ptr = x->get_ptr();
        T *out_ptr = out->get_ptr();
        unsigned int size = out->get_size();

        for (unsigned int i = 0; i < size; i++) {
            out_ptr[i] = alpha * x_ptr[i] + beta * out_ptr[i];
        }
    }
    #if defined(_HAS_CUDA_)
    else {
        add_in_place_device(alpha, x, beta, out);
    }
    #endif
}
template void add_in_place(int alpha, Tensor<int> *x, int beta, Tensor<int> *out);
template void add_in_place(float alpha, Tensor<float> *x, float beta, Tensor<float> *out);
template void add_in_place(double alpha, Tensor<double> *x, double beta, Tensor<double> *out);

#if defined(_HAS_CUDA_)
template <typename T>
void add_in_place_device(T alpha, Tensor<T> *x, T beta, Tensor<T> *out) {


    cudnnErrchk( cudnnAddTensor(
        ::magmadnn::internal::MAGMADNN_SETTINGS->cudnn_handle,
        &alpha,
        x->get_cudnn_tensor_descriptor(),
        x->get_ptr(),
        &beta,
        out->get_cudnn_tensor_descriptor(),
        out->get_ptr()
        ) 
    );
}
#endif

}
}