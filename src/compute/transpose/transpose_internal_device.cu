
#include "compute/transpose/transpose_internal.h"

namespace magmadnn {
namespace internal {
 
template <typename T>
__global__ void kernel_transpose_full_device() {}
 
template <typename T>
void transpose_full_device(Tensor<T> *x, Tensor<T> *out) {

}
template void transpose_full_device(Tensor<int> *x, Tensor<int> *out);
template void transpose_full_device(Tensor<float> *x, Tensor<float> *out);
template void transpose_full_device(Tensor<double> *x, Tensor<double> *out);
 
}   // namespace op
}   // namespace magmadnn
