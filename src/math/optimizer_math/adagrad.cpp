/**
 * @file adagrad.cpp
 * @author Sedrick Keh
 * @version 1.0
 * @date 2019-07-25
 *
 * @copyright Copyright (c) 2019
 */
#include "math/optimizer_math/adagrad.h"

#include <cassert>

namespace magmadnn {
namespace math {

template <typename T>
void adagrad(T learning_rate, Tensor<T> *scaling_tensors, Tensor<T> *grad, Tensor<T> *out) {
    assert(scaling_tensors->get_size() == grad->get_size() && grad->get_size() == out->get_size());
    if (out->get_memory_type() == HOST) {
        T *scaling_tensors_ptr = scaling_tensors->get_ptr();
        T *grad_ptr = grad->get_ptr();
        T *out_ptr = out->get_ptr();
        unsigned int size = out->get_size();

        for (unsigned int i = 0; i < size; i++) {
            scaling_tensors_ptr[i] += (grad_ptr[i] * grad_ptr[i]);
            out_ptr[i] = out_ptr[i] - (learning_rate / sqrt(1e-8 + scaling_tensors_ptr[i])) * grad_ptr[i];
        }
    }
#if defined(_HAS_CUDA_)
    else {
        adagrad_device(learning_rate, scaling_tensors, grad, out);
    }
#endif
}
template void adagrad(int learning_rate, Tensor<int> *scaling_tensors, Tensor<int> *grad, Tensor<int> *out);
template void adagrad(float learning_rate, Tensor<float> *scaling_tensors, Tensor<float> *grad, Tensor<float> *out);
template void adagrad(double learning_rate, Tensor<double> *scaling_tensors, Tensor<double> *grad, Tensor<double> *out);

}  // namespace math
}  // namespace magmadnn
