
#pragma once

#include "tensor/tensor.h"
#include "utilities_internal.h"

namespace magmadnn {
namespace internal {

template <typename T>
void pow_grad(Tensor<T> *x, int power, Tensor<T> *grad, Tensor<T> *out);

#if defined(_HAS_CUDA_)
template <typename T>
void pow_grad_device(Tensor<T> *x, int power, Tensor<T> *grad, Tensor<T> *out);
#endif

}  // namespace internal
}  // namespace magmadnn