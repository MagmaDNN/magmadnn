/**
 * @file rmsprop.cpp
 * @author Sedrick Keh
 * @version 1.0
 * @date 2019-07-25
 *
 * @copyright Copyright (c) 2019
 */
#include "math/optimizer_math/rmsprop.h"

namespace magmadnn {
namespace math {

template <typename T>
void rmsprop(T learning_rate, T decaying_factor, Tensor<T> *decaying_squares_average, Tensor<T> *grad, Tensor<T> *out) {
    assert(decaying_squares_average->get_size() == grad->get_size() && grad->get_size() == out->get_size());
    if (out->get_memory_type() == HOST) {
        T *decaying_squares_average_ptr = decaying_squares_average->get_ptr();
        T *grad_ptr = grad->get_ptr();
        T *out_ptr = out->get_ptr();
        unsigned int size = out->get_size();

        for (unsigned int i = 0; i < size; i++) {
            decaying_squares_average_ptr[i] = (decaying_factor * decaying_squares_average_ptr[i]) +
                                              (1 - decaying_factor) * (grad_ptr[i] * grad_ptr[i]);
            out_ptr[i] = out_ptr[i] - (learning_rate / sqrt(1e-8 + decaying_squares_average_ptr[i])) * grad_ptr[i];
        }
    }
#if defined(_HAS_CUDA_)
    else {
        rmsprop_device(learning_rate, decaying_factor, decaying_squares_average, grad, out);
    }
#endif
}
template void rmsprop(int learning_rate, int decaying_factor, Tensor<int> *decaying_squares_average, Tensor<int> *grad,
                      Tensor<int> *out);
template void rmsprop(float learning_rate, float decaying_factor, Tensor<float> *decaying_squares_average,
                      Tensor<float> *grad, Tensor<float> *out);
template void rmsprop(double learning_rate, double decaying_factor, Tensor<double> *decaying_squares_average,
                      Tensor<double> *grad, Tensor<double> *out);

}  // namespace math
}  // namespace magmadnn
