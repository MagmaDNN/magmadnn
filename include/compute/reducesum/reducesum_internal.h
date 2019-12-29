#pragma once

#include "magmadnn/types.h"
#include "magmadnn/utilities_internal.h"
#include "math/tile.h"
#include "tensor/tensor.h"

#if defined(MAGMADNN_HAVE_CUDA)
#include "cudnn.h"
#include "magma.h"
#endif

namespace magmadnn {
namespace internal {

template <typename T>
void reduce_sum_grad(Tensor<T> *grad, int axis, Tensor<T> *out);

#if defined(MAGMADNN_HAVE_CUDA)
template <typename T>
void reduce_sum_grad_device(Tensor<T> *grad, int axis, Tensor<T> *out);
#endif

}  // namespace internal
}  // namespace magmadnn
