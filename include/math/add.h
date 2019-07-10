/**
 * @file add.h
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-06-24
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "tensor/tensor.h"

namespace magmadnn {
namespace math {

template <typename T>
void add_in_place(T alpha, Tensor<T> *x, T beta, Tensor<T> *out);

#if defined(_HAS_CUDA_)
template <typename T>
void add_in_place_device(T alpha, Tensor<T> *x, T beta, Tensor<T> *out);
#endif

}  // namespace math
}  // namespace magmadnn