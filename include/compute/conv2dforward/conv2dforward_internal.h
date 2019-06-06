
#pragma once

#include "tensor/tensor.h"

namespace magmadnn {
namespace internal {

template <typename T>
void conv2dforward_full(Tensor<T> *in, Tensor<T> *out);


}   // namespace internal
}   // namespace magmadnn