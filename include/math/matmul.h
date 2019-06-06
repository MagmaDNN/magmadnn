/**
 * @file matmul.h
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
void matmul(T alpha, bool trans_A, Tensor<T> *A, bool trans_B, Tensor<T> *B, T beta, Tensor<T> *C);

}
}