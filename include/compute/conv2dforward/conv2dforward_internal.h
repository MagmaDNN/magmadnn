
#pragma once

#include "tensor/tensor.h"

namespace magmadnn {
namespace internal {

template <typename T>
void conv2dforward_full();


#if defined(_HAS_CUDA_)
template <typename T>
void conv2dforward_full_device();
#endif


}   // namespace internal
}   // namespace magmadnn