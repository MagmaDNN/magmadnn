/**
 * @file pow.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-06-10
 *
 * @copyright Copyright (c) 2019
 *
 */
#pragma once

#include <cmath>
#if defined(MAGMADNN_CMAKE_BUILD)
#include "magmadnn/config.h"
#endif
#include "tensor/tensor.h"

namespace magmadnn {
namespace math {

template <typename T>
void pow_cpu(Tensor<T> *x, int power, Tensor<T> *out);

template <typename T>
void pow(Tensor<T> *x, int power, Tensor<T> *out);

#if defined(MAGMADNN_HAVE_CUDA)
template <typename T>
void pow_device(Tensor<T> *x, int power, Tensor<T> *out);

template <typename T>
void pow_device(cudaStream_t custream, Tensor<T> *x, int power, Tensor<T> *out);
#endif

}  // namespace math
}  // namespace magmadnn
