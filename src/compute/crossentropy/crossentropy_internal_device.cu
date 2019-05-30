
#include "compute/crossentropy/crossentropy_internal.h"

namespace magmadnn {
namespace internal {
 
template <typename T>
__global__ void kernel_crossentropy_full_device(T *x, T *y, T *softmax, T *out) {

}

template <typename T>
void crossentropy_full_device(Tensor<T> *x, Tensor<T> *y, Tensor<T> *softmax, Tensor<T> *out) {

}
template void crossentropy_full_device(Tensor<int> *x, Tensor<int> *y, Tensor<int> *softmax, Tensor<int> *out);
template void crossentropy_full_device(Tensor<float> *x, Tensor<float> *y, Tensor<float> *softmax, Tensor<float> *out);
template void crossentropy_full_device(Tensor<double> *x, Tensor<double> *y, Tensor<double> *softmax, Tensor<double> *out);
 
}   // namespace op
}   // namespace magmadnn