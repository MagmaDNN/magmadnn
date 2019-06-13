/**
 * @file pow.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-06-10
 * 
 * @copyright Copyright (c) 2019
 * 
 */
#include "math/pow.h"

#define BLK_SIZE 1024

namespace magmadnn {
namespace math {

template <typename T>
__global__ void kernel_pow_device(T *x, int power, T *out, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < size; i += stride) {
        out[i] = (T) powf((float)x[i], (float)power);
    }
}

template <typename T>
void pow_device(Tensor<T> *x, int power, Tensor<T> *out) {
    unsigned int size = out->get_size();
    kernel_pow_device <<<(size+BLK_SIZE-1)/BLK_SIZE,BLK_SIZE>>> (x->get_ptr(), power, out->get_ptr(), size);
}
template void pow_device(Tensor<int> *x, int power, Tensor<int> *out);
template void pow_device(Tensor<float> *x, int power, Tensor<float> *out);
template void pow_device(Tensor<double> *x, int power, Tensor<double> *out);


}
}

#undef BLK_SIZE
