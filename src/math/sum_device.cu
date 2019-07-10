/**
 * @file sum_device.cu
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-07-10
 *
 * @copyright Copyright (c) 2019
 */
#include "math/sum.h"

namespace magmadnn {
namespace math {

template <typename T>
void kernel_sum_device(T** tensors, T* out, unsigned int size) {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    /* TODO -- implement kernel */
}

template <typename T>
void sum_device(std::vector<Tensor<T>*>& tensors, Tensor<T>* out) {
    /* TODO -- call sum kernel */
}
template void sum_device(std::vector<Tensor<int>*>& tensors, Tensor<int>* out);
template void sum_device(std::vector<Tensor<float>*>& tensors, Tensor<float>* out);
template void sum_device(std::vector<Tensor<double>*>& tensors, Tensor<double>* out);

}  // namespace math
}  // namespace magmadnn