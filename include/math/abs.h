#pragma once

#include <cmath>
#include "tensor/tensor.h"
#include "utilities_internal.h"

namespace magmadnn{
namespace math{
template <typename T>
void abs(Tensor<T>* x, Tensor<T>* out);

#if defined(_HAS_CUDA_)
template <typename T>
void abs_device(Tensor<T>* x, Tensor<T>* out);
#endif

}  //  namespace math
} //  namespace magmadnn