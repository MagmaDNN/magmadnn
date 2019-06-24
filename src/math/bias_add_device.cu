/**
* @file bias_add_device.cu
* @author Daniel Nichols
* @version 0.1
* @date 2019-06-23
* 
* @copyright Copyright (c) 2019
*/
#include "math/bias_add.h"

#define BLK_SIZE 1024

namespace magmadnn {
namespace math {

template <typename T>
__global__ void kernel_bias_add_device(T *x, T *bias, T *out, unsigned int x_rows, unsigned int x_cols) {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < x_cols*x_rows; i += stride) {
        out[i] = x[i] + bias[i / x_cols];
    }
}

template <typename T>
void bias_add_device(Tensor<T> *x, Tensor<T> *bias, Tensor<T> *out) {
    unsigned int x_rows = x->get_shape(0);
    unsigned int x_cols = x->get_shape(1);

    kernel_bias_add_device<<<(x_rows*x_cols+BLK_SIZE-1)/BLK_SIZE,BLK_SIZE>>>(x->get_ptr(), bias->get_ptr(), out->get_ptr(), x_rows, x_cols);
}
template void bias_add_device(Tensor<int> *x, Tensor<int> *bias, Tensor<int> *out);
template void bias_add_device(Tensor<float> *x, Tensor<float> *bias, Tensor<float> *out);
template void bias_add_device(Tensor<double> *x, Tensor<double> *bias, Tensor<double> *out);
 
}
}

#undef BLK_SIZE
