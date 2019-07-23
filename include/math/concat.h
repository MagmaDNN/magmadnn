/**
 * @file concat.h
 * @author Sedrick Keh
 * @version 0.1
 * @date 2019-06-14
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "tensor/tensor.h"

namespace magmadnn {
namespace math {

/** Concatenates A and B along axis and puts output into C
 * A, B, C should have at most 1 different dimension
 * @param A
 * @param B
 * @param C axis dim size should equal sum of A and B axis dim size
 */
template <typename T>
void concat(const Tensor &A, const Tensor &B, Tensor &C, index_t axis);

}  // namespace math
}  // namespace magmadnn