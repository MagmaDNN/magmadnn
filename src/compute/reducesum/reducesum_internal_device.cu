
#include "compute/reducesum/reducesum_internal.h"

namespace magmadnn {
namespace internal {

template <typename T>
__global__ void kernel_tensor_reducesum_full_device(T *arr, T *axes, unsigned int n_axes, int axis, T *out) {
    /* TODO */
}
 
template <typename T>
void tensor_reducesum_full_device(Tensor<T> *x, unsigned int axis, Tensor<T> *out) {
    /* TODO */
}
template void tensor_reducesum_full_device(Tensor<int> *x, unsigned int axis, Tensor<int> *out);
template void tensor_reducesum_full_device(Tensor<float> *x, unsigned int axis, Tensor<float> *out);
template void tensor_reducesum_full_device(Tensor<double> *x, unsigned int axis, Tensor<double> *out);


template <typename T>
__global__ void kernel_reducesum_full_device() {
    /* TODOT */
}
 
template <typename T>
void reducesum_full_device(Tensor<T> *x, Tensor<T> *out) {
    /* TODO */
}
template void reducesum_full_device(Tensor<int> *x, Tensor<int> *out);
template void reducesum_full_device(Tensor<float> *x, Tensor<float> *out);
template void reducesum_full_device(Tensor<double> *x, Tensor<double> *out);
 
}   // namespace op
}   // namespace magmadnn