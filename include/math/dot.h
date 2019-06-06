/**
 * @file dot.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-06-06
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once
#include "tensor/tensor.h"

namespace magmadnn {
namespace math {

template <typename T>
void dot(T alpha, Tensor<T> *a, T beta, Tensor<T> *b, Tensor<T> *out);

}
}