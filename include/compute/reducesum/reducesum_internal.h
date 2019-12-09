
#pragma once

// #include "cblas.h"
#include "math/tile.h"
#include "tensor/tensor.h"
#include "types.h"
#include "utilities_internal.h"
#if defined(_HAS_CUDA_)
#include "cudnn.h"
#include "magma.h"
#endif

namespace magmadnn {
namespace internal {

template <typename T>
void reduce_sum_grad(Tensor<T> *grad, int axis, Tensor<T> *out);

#if defined(_HAS_CUDA_)
template <typename T>
void reduce_sum_grad_device(Tensor<T> *grad, int axis, Tensor<T> *out);
#endif

}  // namespace internal
}  // namespace magmadnn
