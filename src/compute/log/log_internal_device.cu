#include "compute/log/log_internal.h"

#define BLK_SIZE 1024

namespace magmadnn {
namespace internal {

template <typename T>
__global__ void kernel_log_full_device(T *x, T *out, unsigned int size) {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < size; i += stride) {
        out[i] = log(x[i]);
    }
}
template <>
__global__ void kernel_log_full_device(int *x, int *out, unsigned int size) {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < size; i += stride) {
        out[i] = (int) log((float) x[i]);
    }
}

template <typename T>
void log_full_device(Tensor<T> *x, Tensor<T> *out) {
    unsigned int size = x->get_size();
    kernel_log_full_device<<<(size + BLK_SIZE - 1) / BLK_SIZE, BLK_SIZE>>>(x->get_ptr(), out->get_ptr(), size);
}
template void log_full_device(Tensor<int> *x, Tensor<int> *out);
template void log_full_device(Tensor<float> *x, Tensor<float> *out);
template void log_full_device(Tensor<double> *x, Tensor<double> *out);

template <typename T>
__global__ void kernel_log_grad_device(T *x, T *grad, T *out, unsigned int size) {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < size; i += stride) {
        out[i] = grad[i] / x[i];
    }
}

template <typename T>
void log_grad_device(Tensor<T> *x, Tensor<T> *grad, Tensor<T> *out) {
    unsigned int size = x->get_size();
    kernel_log_grad_device<<<(size + BLK_SIZE - 1) / BLK_SIZE, BLK_SIZE>>>(x->get_ptr(), grad->get_ptr(),
                                                                           out->get_ptr(), size);
}
template void log_grad_device(Tensor<int> *x, Tensor<int> *grad, Tensor<int> *out);
template void log_grad_device(Tensor<float> *x, Tensor<float> *grad, Tensor<float> *out);
template void log_grad_device(Tensor<double> *x, Tensor<double> *grad, Tensor<double> *out);

}  // namespace internal
}  // namespace magmadnn

#undef BLK_SIZE
