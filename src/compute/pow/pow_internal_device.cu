#include "magmadnn/math.h"
#include "compute/pow/pow_internal.h"

#define BLK_SIZE 1024

namespace magmadnn {
namespace internal {

template <typename T>
__global__ void kernel_pow_grad_device(T *x, int power, T *grad, T *out, bool grad_is_scalar, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < size; i += stride) {
        out[i] = grad[(grad_is_scalar) ? 0 : i] * ((T) power) * powf(x[i], power - 1);
    }
}

template <>
__global__ void kernel_pow_grad_device(int *x, int power, int *grad, int *out, bool grad_is_scalar, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < size; i += stride) {
        out[i] = grad[(grad_is_scalar) ? 0 : i] * ((int) power) * ((int) powf((float) x[i], power - 1));
    }
}

template <typename T>
void pow_grad_device(Tensor<T> *x, int power, Tensor<T> *grad, Tensor<T> *out) {
    unsigned int size = out->get_size();
    kernel_pow_grad_device<<<(size + BLK_SIZE - 1) / BLK_SIZE, BLK_SIZE>>>(
        x->get_ptr(), power, grad->get_ptr(), out->get_ptr(), (grad->get_size() == 1), size);
}
template void pow_grad_device(Tensor<int> *x, int power, Tensor<int> *input, Tensor<int> *out);
template void pow_grad_device(Tensor<float> *x, int power, Tensor<float> *input, Tensor<float> *out);
template void pow_grad_device(Tensor<double> *x, int power, Tensor<double> *input, Tensor<double> *out);

template <typename T>
void pow_grad_device(cudaStream_t custream, Tensor<T> *x, int power, Tensor<T> *grad, Tensor<T> *out) {

   unsigned int size = out->get_size();
   const auto grid_dim = ceildiv(size, BLK_SIZE);

   kernel_pow_grad_device
      <<<grid_dim, BLK_SIZE, 0, custream>>>
      (x->get_ptr(), power, grad->get_ptr(), out->get_ptr(),
       (grad->get_size() == 1), size);
}
template void pow_grad_device(cudaStream_t custream, Tensor<int> *x, int power, Tensor<int> *input, Tensor<int> *out);
template void pow_grad_device(cudaStream_t custream, Tensor<float> *x, int power, Tensor<float> *input, Tensor<float> *out);
template void pow_grad_device(cudaStream_t custream, Tensor<double> *x, int power, Tensor<double> *input, Tensor<double> *out);
   
}  // namespace internal
}  // namespace magmadnn

#undef BLK_SIZE
