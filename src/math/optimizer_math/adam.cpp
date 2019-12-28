/**
 * @file adam.cpp
 * @author Sedrick Keh
 * @version 1.0
 * @date 2019-07-25
 *
 * @copyright Copyright (c) 2019
 */
#include "math/optimizer_math/adam.h"

#include <cassert>

namespace magmadnn {
namespace math {

template <typename T>
void adam(T learning_rate, T beta1, T beta2, T running_beta1, T running_beta2, Tensor<T> *first_moment,
          Tensor<T> *second_moment, Tensor<T> *grad, Tensor<T> *out) {
    assert(first_moment->get_size() == second_moment->get_size() && second_moment->get_size() == grad->get_size() &&
           grad->get_size() == out->get_size());
    if (out->get_memory_type() == HOST) {
        T *first_moment_ptr = first_moment->get_ptr();
        T *second_moment_ptr = second_moment->get_ptr();
        T *grad_ptr = grad->get_ptr();
        T *out_ptr = out->get_ptr();
        unsigned int size = out->get_size();

        for (unsigned int i = 0; i < size; i++) {
            first_moment_ptr[i] = (beta1 * first_moment_ptr[i]) + (1 - beta1) * (grad_ptr[i]);
            second_moment_ptr[i] = (beta2 * second_moment_ptr[i]) + (1 - beta2) * (grad_ptr[i] * grad_ptr[i]);
            T m_temp = first_moment_ptr[i] / (1 - running_beta1);
            T v_temp = second_moment_ptr[i] / (1 - running_beta2);
            out_ptr[i] = out_ptr[i] - (learning_rate / (sqrt(v_temp) + 1e-8)) * m_temp;
        }
    }
#if defined(_HAS_CUDA_)
    else {
        adam_device(learning_rate, beta1, beta2, running_beta1, running_beta2, first_moment, second_moment, grad, out);
    }
#endif
}
template void adam(int learning_rate, int beta1, int beta2, int running_beta1, int running_beta2,
                   Tensor<int> *first_moment, Tensor<int> *second_moment, Tensor<int> *grad, Tensor<int> *out);
template void adam(float learning_rate, float beta1, float beta2, float running_beta1, float running_beta2,
                   Tensor<float> *first_moment, Tensor<float> *second_moment, Tensor<float> *grad, Tensor<float> *out);
template void adam(double learning_rate, double beta1, double beta2, double running_beta1, double running_beta2,
                   Tensor<double> *first_moment, Tensor<double> *second_moment, Tensor<double> *grad,
                   Tensor<double> *out);
}  // namespace math
}  // namespace magmadnn
