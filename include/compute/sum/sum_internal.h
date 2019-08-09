/**
 * @file sum_internal.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-20
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include <vector>
#include "tensor/tensor.h"

namespace magmadnn {
namespace internal {

/**
 * @tparam T
 * @param vals
 */
void sum_full(const std::vector<std::reference_wrapper<const Tensor>> &vals, Tensor &out);

#if defined(_HAS_CUDA_)
template <typename T>
void sum_full_device(const std::vector<std::reference_wrapper<const Tensor>> &vals, Tensor &out);
#endif

}  // namespace internal
}  // namespace magmadnn