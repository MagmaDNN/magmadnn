
#include "compute/scalarproduct/scalarproduct_internal.h"

namespace magmadnn {
namespace internal {

template <typename T>
__global__ void kernel_scalarproduct_full_device(T alpha, T *arr, T *out, unsigned int arr_size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < arr_size; i += stride) {
        out[i] = alpha * arr[i];
    }
}

template <typename T>
void scalarproduct_full_device(T alpha, Tensor<T> *x, Tensor<T> *out) {
    kernel_scalarproduct_full_device<<<1, x->get_size()>>>(alpha, x->get_ptr(), out->get_ptr(), x->get_size());
}
template void scalarproduct_full_device(int alpha, Tensor<int> *x, Tensor<int> *out);
template void scalarproduct_full_device(float alpha, Tensor<float> *x, Tensor<float> *out);
template void scalarproduct_full_device(double alpha, Tensor<double> *x, Tensor<double> *out);

}  // namespace internal
}  // namespace magmadnn
