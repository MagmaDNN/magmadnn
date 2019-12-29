/**
 * @file negate.h
 * @author Daniel Nichols
 * @author Florent Lopez
 * @version 0.1
 * @date 2019-06-13
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "tensor/tensor.h"

namespace magmadnn {
namespace math {

template <typename T>
void negate(Tensor<T> *x, Tensor<T> *out);
}
}  // namespace magmadnn
