
#include "compute/reducesum/reducesum_internal.h"

namespace magmadnn {
namespace internal {
 
template <typename T>
__global__ void kernel_reducesum_full_device() {}
 
template <typename T>
void reducesum_full_device() {}
template void reducesum_full_device();
template void reducesum_full_device();
template void reducesum_full_device();
 
}   // namespace op
}   // namespace magmadnn