
#include "compute/conv2dforward/conv2dforward_internal.h"

namespace magmadnn {
namespace internal {
 
template <typename T>
__global__ void kernel_conv2dforward_full_device() {}
 
template <typename T>
void conv2dforward_full_device() {}
template void conv2dforward_full_device();
template void conv2dforward_full_device();
template void conv2dforward_full_device();
 
}   // namespace op
}   // namespace magmadnn