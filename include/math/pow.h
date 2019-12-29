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
#include "tensor/tensor.h"

namespace magmadnn {
namespace math {

template <typename T>
void pow(Tensor<T> *x, int power, Tensor<T> *out);

#if defined(MAGMADNN_HAVE_CUDA)
template <typename T>
void pow_device(Tensor<T> *x, int power, Tensor<T> *out);
#endif

}  // namespace math
}  // namespace magmadnn
