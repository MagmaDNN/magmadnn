/**
 * @file tanh_internal.h
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-23
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once
#include <math.h>
#include "tensor/tensor.h"


namespace skepsi {
namespace internal {

/** Computes the element-wise tanh function.
 * @tparam T 
 * @param x 
 */
template <typename T>
void tanh_full(Tensor<T> *x);

#if defined(_HAS_CUDA_)
/** Computes the tanh function element-wise on the tensor x
 * @tparam T 
 * @param x 
 */
template <typename T>
void tanh_full_device(Tensor<T> *x); 
#endif

}   // namespace internal
}   // namespace skepsi