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
#include "math/matmul.h"

namespace magmadnn {
namespace math {

template <typename T>
void dot(Tensor<T> *A, Tensor<T> *B, Tensor<T> *out);

template <typename T>
void dot(T alpha, Tensor<T> *A, Tensor<T> *B, T beta, Tensor<T> *out);

template <typename T>
void dot(T alpha, bool trans_A, Tensor<T> *A, bool trans_B, Tensor<T> *B, T beta, Tensor<T> *out);

}
}