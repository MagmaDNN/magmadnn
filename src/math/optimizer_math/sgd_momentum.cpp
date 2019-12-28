/**
 * @file sgd_momentum.cpp
 * @author Sedrick Keh
 * @version 1.0
 * @date 2019-07-25
 *
 * @copyright Copyright (c) 2019
 */
#include "math/optimizer_math/sgd_momentum.h"

#include <cassert>

namespace magmadnn {
namespace math {

template <typename T>
void sgd_momentum(T learning_rate, T momentum, Tensor<T> *prev, Tensor<T> *grad, Tensor<T> *out) {
    assert(prev->get_size() == grad->get_size() && grad->get_size() == out->get_size());
    if (out->get_memory_type() == HOST) {
        T *prev_ptr = prev->get_ptr();
        T *grad_ptr = grad->get_ptr();
        T *out_ptr = out->get_ptr();
        unsigned int size = out->get_size();

        for (unsigned int i = 0; i < size; i++) {
            prev_ptr[i] = momentum * prev_ptr[i] + (1 - momentum) * grad_ptr[i];
            out_ptr[i] = out_ptr[i] - learning_rate * prev_ptr[i];
        }
    }
#if defined(_HAS_CUDA_)
    else {
        sgd_momentum_device(learning_rate, momentum, prev, grad, out);
    }
#endif
}
template void sgd_momentum(int learning_rate, int momentum, Tensor<int> *prev, Tensor<int> *grad, Tensor<int> *out);
template void sgd_momentum(float learning_rate, float momentum, Tensor<float> *prev, Tensor<float> *grad,
                           Tensor<float> *out);
template void sgd_momentum(double learning_rate, double momentum, Tensor<double> *prev, Tensor<double> *grad,
                           Tensor<double> *out);
}  // namespace math
}  // namespace magmadnn
