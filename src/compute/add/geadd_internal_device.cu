/**
 * @file geadd_internal_device.cu
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-02-22
 *
 * @copyright Copyright (c) 2019
 */

#include "tensor/tensor.h"

#define BLK_SIZE 1024

namespace magmadnn {
namespace internal {

template <typename T>
__global__ void kernel_geadd_full_device(T alpha, T *A, T beta, T *B, T *C, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < size; i += stride) {
        C[i] = alpha * A[i] + beta * B[i];
    }
}

template <typename T>
void geadd_full_device(T alpha, Tensor<T> *A, T beta, Tensor<T> *B, Tensor<T> *C) {
    unsigned int size = C->get_size();
    kernel_geadd_full_device<<<(size + BLK_SIZE - 1) / BLK_SIZE, BLK_SIZE>>>(alpha, A->get_ptr(), beta, B->get_ptr(),
                                                                             C->get_ptr(), size);
}
template void geadd_full_device(int alpha, Tensor<int> *A, int beta, Tensor<int> *B, Tensor<int> *C);
template void geadd_full_device(float alpha, Tensor<float> *A, float beta, Tensor<float> *B, Tensor<float> *C);
template void geadd_full_device(double alpha, Tensor<double> *A, double beta, Tensor<double> *B, Tensor<double> *C);

template <typename T>
__global__ void kernel_tensor_scalar_add_full_device(T alpha, T *x, T *out, unsigned int arr_size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < arr_size; i += stride) {
        out[i] = alpha + x[i];
    }
}

template <typename T>
void tensor_scalar_add_full_device(T alpha, Tensor<T> *x, Tensor<T> *out) {
    unsigned int size = out->get_size();
    kernel_tensor_scalar_add_full_device<<<(size + BLK_SIZE - 1) / BLK_SIZE, BLK_SIZE>>>(alpha, x->get_ptr(),
                                                                                         out->get_ptr(), size);
}
template void tensor_scalar_add_full_device(int alpha, Tensor<int> *x, Tensor<int> *out);
template void tensor_scalar_add_full_device(float alpha, Tensor<float> *x, Tensor<float> *out);
template void tensor_scalar_add_full_device(double alpha, Tensor<double> *x, Tensor<double> *out);

}  // namespace internal
}  // namespace magmadnn

#undef BLK_SIZE
