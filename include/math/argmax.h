/**
 * @file argmax.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-06-11
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "tensor/tensor.h"

namespace magmadnn {
namespace math {

/** Returns the argmax along the specified axis.
 * @tparam T
 * @param x
 * @param axis
 * @param out
 */
template <typename T>
void argmax(Tensor<T> *x, int axis, Tensor<T> *out);

}  // namespace math
}  // namespace magmadnn