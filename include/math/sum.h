/**
 * @file sum.h
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-07-10
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "tensor/tensor.h"
#include "utilities_internal.h"

namespace magmadnn {
namespace math {

template <typename T>
void sum(const std::vector<std::reference_wrapper<const Tensor>>& tensors, Tensor& out);

#if defined(_HAS_CUDA_)
template <typename T>
void sum_device(const std::vector<std::reference_wrapper<const Tensor>>& tensors, Tensor& out);
#endif

}  // namespace math
}  // namespace magmadnn