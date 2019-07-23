/**
 * @file crossentropy.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-06-12
 *
 * @copyright Copyright (c) 2019
 *
 */
#pragma once

#include "tensor/tensor.h"
#include "utilities_internal.h"

namespace magmadnn {
namespace math {

/** Computes the cross entropy of predicted and ground_truth into out.
 * @tparam T
 * @param predicted
 * @param ground_truth
 * @param out scalar tensor
 */
template <typename T>
void crossentropy(const Tensor &predicted, const Tensor &ground_truth, Tensor &out);

#if defined(_HAS_CUDA_)
template <typename T>
void crossentropy_device(const Tensor &predicted, const Tensor &ground_truth, Tensor &out);
#endif

}  // namespace math
}  // namespace magmadnn