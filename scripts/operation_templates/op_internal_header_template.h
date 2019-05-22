
#pragma once

#include "tensor/tensor.h"

namespace magmadnn {
namespace internal {

template <typename T>
void <#OPERATION_NAME_LOWER#>_full();


#if defined(_HAS_CUDA_)
template <typename T>
void <#OPERATION_NAME_LOWER#>_full_device();
#endif


}   // namespace internal
}   // namespace magmadnn