/**
 * @file sum.h
 * @author Daniel Nichols
 * @author Florent Lopez
 * @version 1.0
 * @date 2019-07-10
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "tensor/tensor.h"

namespace magmadnn {
namespace math {

template <typename T>
void sum(const std::vector<Tensor<T>*>& tensors, Tensor<T>* out);

#if defined(MAGMADNN_HAVE_CUDA)
template <typename T>
void sum_device(const std::vector<Tensor<T>*>& tensors, Tensor<T>* out);
#endif

}  // namespace math
}  // namespace magmadnn
