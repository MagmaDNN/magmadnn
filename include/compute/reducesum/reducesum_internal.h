
#pragma once

#include "tensor/tensor.h"
#include "math/tile.h"
#include "utilities_internal.h"
#include "cblas.h"
#include "types.h"
#if defined(_HAS_CUDA_)
#include "magma.h"
#include "cudnn_v7.h"
#endif

namespace magmadnn {
namespace internal {

template <typename T>
void reduce_sum_grad(Tensor<T> *grad, int axis, Tensor<T> *out);

#if defined(_HAS_CUDA_)
template <typename T>
void reduce_sum_grad_device(Tensor<T> *grad, int axis, Tensor<T> *out);
#endif

}   // namespace internal
}   // namespace magmadnn