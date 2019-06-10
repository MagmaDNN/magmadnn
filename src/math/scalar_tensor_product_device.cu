/**
 * @file scalar_tensor_product_device.cu
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-06-07
 * 
 * @copyright Copyright (c) 2019
 */
 #include "math/scalar_tensor_product.h"

namespace magmadnn {
namespace math {

template <typename T>
__global__ void kernel_scalar_tensor_product_device(T scalar, T *x, T *out, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < size; i += stride) {
        out[i] = scalar * x[i];
    }
}

template <typename T>
void scalar_tensor_product_device(T scalar, Tensor<T> *x, Tensor<T> *out) {
    unsigned int size = out->get_size();
    kernel_scalar_tensor_product_device <<< 1, size >>> (scalar, x->get_ptr(), out->get_ptr(), size);
}
template void scalar_tensor_product_device(int scalar, Tensor<int> *x, Tensor<int> *out);
template void scalar_tensor_product_device(float scalar, Tensor<float> *x, Tensor<float> *out);
template void scalar_tensor_product_device(double scalar, Tensor<double> *x, Tensor<double> *out);

}
}