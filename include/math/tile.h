/**
 * @file tile.h
 * @author Sedrick Keh
 * @version 0.1
 * @date 2019-06-17
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "tensor/tensor.h"

namespace magmadnn {
namespace math {

/** Tiles tensor A along axis t times, output into B
 * A, B should have at most 1 different dimension
 * @param A
 * @param B axis dim should equal A axis dim * t
 */
template <typename T>
void tile(Tensor<T> *A, Tensor<T> *B, unsigned int t, unsigned int axis);

}  // namespace math
}  // namespace magmadnn