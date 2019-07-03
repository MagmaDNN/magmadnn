/**
 * @file product.h
 * @author Sedrick Keh
 * @version 0.1
 * @date 2019-07-02
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "tensor/tensor.h"

namespace magmadnn {
namespace math {

/** (In-place) Copies and returns the element-wise product of A and B into B.
 * @tparam T 
 * @param A 
 * @param B should have same shape as A 
 */
template <typename T>
void product(Tensor<T> *A, Tensor<T> *B);

/** Copies and returns the element-wise product of A and B into C.
 * @tparam T 
 * @param A 
 * @param B should have same shape as A 
 * @param C should have same shape as A and B
 */
template <typename T>
void product(Tensor<T> *A, Tensor<T> *B, Tensor<T> *C);
}
}