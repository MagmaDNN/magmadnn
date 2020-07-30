/**
 * @file sgd_momentum.h
 * @author Sedrick Keh
 * @version 1.0
 * @date 2019-07-25
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#if defined(MAGMADNN_CMAKE_BUILD)
#include "magmadnn/config.h"
#endif
#include "tensor/tensor.h"

namespace magmadnn {
namespace math {

template <typename T>
void sgd_momentum_cpu(
      T learning_rate, T momentum, Tensor<T> *prev, Tensor<T> *grad,
      Tensor<T> *out);

template <typename T>
void sgd_momentum_cpu(
      T learning_rate, T momentum, Tensor<T> *prev, Tensor<T> *grad,
      std::vector<int> *idxs, Tensor<T> *out);   
   
/** Do the SGD update with momentum.
 * @tparam T
 * @param learning_rate
 * @param momentum
 * @param prev
 * @param grad
 * @param out
 */
template <typename T>
void sgd_momentum(T learning_rate, T momentum, Tensor<T> *prev, Tensor<T> *grad, Tensor<T> *out);

#if defined(MAGMADNN_HAVE_CUDA)
template <typename T>
void sgd_momentum_device(T learning_rate, T momentum, Tensor<T> *prev, Tensor<T> *grad, Tensor<T> *out);

template <typename T>
void sgd_momentum_device(cudaStream_t custream, T learning_rate, T momentum, Tensor<T> *prev, Tensor<T> *grad, Tensor<T> *out);
#endif

}  // namespace math
}  // namespace magmadnn
