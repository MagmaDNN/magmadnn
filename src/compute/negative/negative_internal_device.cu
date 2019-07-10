#include "compute/negative/negative_internal.h"

#define BLK_SIZE 1024

namespace magmadnn {
namespace internal {

template <typename T>
__global__ void kernel_negative_full_device(T *x, T *out, unsigned int size) {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < size; i += stride) {
        out[i] = -x[i];
    }
}

template <typename T>
void negative_full_device(Tensor<T> *x, Tensor<T> *out) {
    unsigned int size = out->get_size();
    kernel_negative_full_device<<<(size + BLK_SIZE - 1) / BLK_SIZE, BLK_SIZE>>>(x->get_ptr(), out->get_ptr(), size);
}
template void negative_full_device(Tensor<int> *x, Tensor<int> *out);
template void negative_full_device(Tensor<float> *x, Tensor<float> *out);
template void negative_full_device(Tensor<double> *x, Tensor<double> *out);

}  // namespace internal
}  // namespace magmadnn

#undef BLK_SIZE
