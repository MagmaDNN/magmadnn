/**
 * @file rmsprop.h
 * @author Sedrick Keh
 * @author Florent Lopez
 * @version 1.0
 * @date 2019-07-25
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "tensor/tensor.h"

namespace magmadnn {
namespace math {

/** Do the RMSProp gradient update.
 * @tparam T
 * @param learning_rate
 * @param decaying_factor
 * @param decaying_squares_average
 * @param grad
 * @param out
 */
template <typename T>
void rmsprop(T learning_rate, T decaying_factor, Tensor<T> *decaying_squares_average, Tensor<T> *grad, Tensor<T> *out);

#if defined(MAGMADNN_HAVE_CUDA)
template <typename T>
void rmsprop_device(T learning_rate, T decaying_factor, Tensor<T> *decaying_squares_average, Tensor<T> *grad,
                    Tensor<T> *out);
#endif

}  // namespace math
}  // namespace magmadnn
