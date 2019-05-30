/**
 * @file gradientdescent_internal.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-30
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "tensor/tensor.h"

namespace magmadnn {
namespace internal {

template <typename T>
magmadnn_error_t gradientdescent_update_internal(Tensor<T> *var, Tensor<T> *grad, T learning_rate);

#if defined(_HAS_CUDA_)
template <typename T>
magmadnn_error_t gradientdescent_update_internal_device(Tensor<T> *var, Tensor<T> *grad, T learning_rate);
#endif

}   // namespace internal
}   // namespace magmadnn