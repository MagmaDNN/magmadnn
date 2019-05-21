/**
 * @file relu_internal_device.cu
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-01
 * 
 * @copyright Copyright (c) 2019
 */
 #include "compute/relu_internal.h"

namespace magmadnn {
namespace internal {

template <typename T>
__global__ void kernel_relu_full_device(unsigned int size, T *arr, T *out) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    
    for (unsigned int i = idx; i < size; i += stride) {
        if (arr[i] < 0) out[i] = 0;
        else out[i] = arr[i];
    }
}

template <typename T>
void relu_full_device(Tensor<T> *x, Tensor<T> *out) {
    kernel_relu_full_device <<< x->get_size(), 1 >>> (x->get_size(), x->get_ptr(), out->get_ptr());
}
template void relu_full_device(Tensor<int> *x, Tensor<int> *out);
template void relu_full_device(Tensor<float> *x, Tensor<float> *out);
template void relu_full_device(Tensor<double> *x, Tensor<double> *out);

}   // internal
}   // magmadnn