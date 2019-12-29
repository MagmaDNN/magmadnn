/**
 * @file adagrad.h
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

/** Do the Adagrad gradient update.
 * @tparam T
 * @param learning_rate
 * @param scaling_tensors
 * @param grad
 * @param out
 */
template <typename T>
void adagrad(T learning_rate, Tensor<T> *scaling_tensors,
             Tensor<T> *grad, Tensor<T> *out);

#if defined(MAGMADNN_HAVE_CUDA)
template <typename T>
void adagrad_device(T learning_rate, Tensor<T> *scaling_tensors,
                    Tensor<T> *grad, Tensor<T> *out);
#endif

}  // namespace math
}  // namespace magmadnn
