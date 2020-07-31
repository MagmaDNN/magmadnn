/**
 * @file adam.h
 * @author Sedrick Keh
 * @version 1.0
 * @date 2019-07-25
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "tensor/tensor.h"

namespace magmadnn {
namespace math {

/** Do the Adam gradient update.
 * @tparam T
 * @param learning_rate
 * @param beta1
 * @param beta2
 * @param running_beta1
 * @param running_beta2
 * @param first_moment
 * @param second_moment
 * @param grad
 * @param out
 */
template <typename T>
void adam(T learning_rate, T beta1, T beta2, T running_beta1, T running_beta2, Tensor<T> *first_moment,
          Tensor<T> *second_moment, Tensor<T> *grad, Tensor<T> *out);

#if defined(MAGMADNN_HAVE_CUDA)
template <typename T>
void adam_device(T learning_rate, T beta1, T beta2, T running_beta1, T running_beta2, Tensor<T> *first_moment,
                 Tensor<T> *second_moment, Tensor<T> *grad, Tensor<T> *out);
#endif

}  // namespace math
}  // namespace magmadnn
