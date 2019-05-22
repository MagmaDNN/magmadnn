
#include "compute/<#OPERATION_NAME_LOWER#>/<#OPERATION_NAME_LOWER#>_internal.h"

namespace magmadnn {
namespace internal {
 
template <typename T>
__global__ void kernel_<#OPERATION_NAME_LOWER#>_full_device() {}
 
template <typename T>
void <#OPERATION_NAME_LOWER#>_full_device() {}
template void <#OPERATION_NAME_LOWER#>_full_device();
template void <#OPERATION_NAME_LOWER#>_full_device();
template void <#OPERATION_NAME_LOWER#>_full_device();
 
}   // namespace op
}   // namespace magmadnn