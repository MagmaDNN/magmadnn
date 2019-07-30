#include "math/abs.h"

#define BLK_SIZE 1024

namespace magmadnn {
namespace math {

template <typename T>
__global__ void kernel_abs_device(T *x, T *out, unsigned size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = idx; i < size; i += stride) {
        out[i] = std::abs(x[i]);
    }
}

template <typename T>
void abs_device(Tensor<T>* x, Tensor<T>* out) {
    unsigned size = out->get_size();
    kernel_abs_device<<<(size + BLK_SIZE - 1) / BLK_SIZE, BLK_SIZE>>>(x->get_ptr(), out->get_ptr(), size);
}
template void abs_device(Tensor<int>* x, Tensor<int>* out);
template void abs_device(Tensor<float>* x, Tensor<float>* out);
template void abs_device(Tensor<double>* x, Tensor<double>* out);

}  // namespace math
}  // namespace magmadnn

#undef BLK_SIZE