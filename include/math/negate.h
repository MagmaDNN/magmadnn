/**
 * @file negate.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-06-13
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "tensor/tensor.h"
#include "utilities_internal.h"

namespace magmadnn {
namespace math {

template <typename T>
void negate(Tensor<T> *x, Tensor<T> *out);

}
}