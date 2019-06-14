/**
 * @file scalar_tensor_product.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-06-07
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "tensor/tensor.h"

namespace magmadnn {
namespace math {

template <typename T>
void concat(Tensor<T> *A, Tensor<T> *B, Tensor<T> *C, unsigned int axis);

}
}