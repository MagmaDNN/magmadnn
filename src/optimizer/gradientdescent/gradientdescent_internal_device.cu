/**
 * @file gradientdescent_internal.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-30
 *
 * @copyright Copyright (c) 2019
 */
#include "optimizer/gradientdescent/gradientdescent_internal.h"

#define BLK_SIZE 1024

namespace magmadnn {
namespace internal {

template <typename T>
__global__ void kernel_gradientdescent_update_internal_device(T *var, T *grad, T learning_rate, unsigned int size) {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;

    for (unsigned int i = idx; i < size; i += stride) {
        var[i] -= learning_rate * grad[i];
    }
}

template <typename T>
magmadnn_error_t gradientdescent_update_internal_device(Tensor<T> *var, Tensor<T> *grad, T learning_rate) {
    magmadnn_error_t err = (magmadnn_error_t) 0;

    unsigned int size = var->get_size();
    kernel_gradientdescent_update_internal_device<<<(size + BLK_SIZE - 1) / BLK_SIZE, BLK_SIZE>>>(
        var->get_ptr(), grad->get_ptr(), learning_rate, size);

    return (magmadnn_error_t) err;
}
template magmadnn_error_t gradientdescent_update_internal_device(Tensor<int> *var, Tensor<int> *grad,
                                                                 int learning_rate);
template magmadnn_error_t gradientdescent_update_internal_device(Tensor<float> *var, Tensor<float> *grad,
                                                                 float learning_rate);
template magmadnn_error_t gradientdescent_update_internal_device(Tensor<double> *var, Tensor<double> *grad,
                                                                 double learning_rate);

}  // namespace internal
}  // namespace magmadnn

#undef BLK_SIZE
