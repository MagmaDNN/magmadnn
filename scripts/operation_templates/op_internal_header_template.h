
#pragma once

#include "tensor/tensor.h"

namespace magmadnn {
namespace internal {

template <typename T>
void <#OPERATION_NAME_LOWER#>_full(Tensor<T> *input, Tensor<T> *out);


#if defined(_HAS_CUDA_)
template <typename T>
void <#OPERATION_NAME_LOWER#>_full_device(Tensor<T> *input, Tensor<T> *out);
#endif

template <typename T>
void <#OPERATION_NAME_LOWER#>_grad(Tensor<T> *grad, Tensor<T> *out);


#if defined(_HAS_CUDA_)
template <typename T>
void <#OPERATION_NAME_LOWER#>_grad_device(Tensor<T> *grad, Tensor<T> *out);
#endif


}   // namespace internal
}   // namespace magmadnn