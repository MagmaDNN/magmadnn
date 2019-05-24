
#pragma once

#include "tensor/tensor.h"

namespace magmadnn {
namespace internal {

template <typename T>
void reducesum_full();


#if defined(_HAS_CUDA_)
template <typename T>
void reducesum_full_device();
#endif


}   // namespace internal
}   // namespace magmadnn