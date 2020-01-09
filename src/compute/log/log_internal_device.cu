#include "compute/log/log_internal.h"

#include "magmadnn/math.h"

#define BLK_SIZE 1024

namespace magmadnn {
namespace internal {

template <typename T>
__global__ void kernel_log_full_device(T *x, T *out, unsigned int size, T epsilon) {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < size; i += stride) {
        out[i] = log(x[i] + epsilon);
    }
}
template <>
__global__ void kernel_log_full_device(int *x, int *out, unsigned int size, int epsilon) {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < size; i += stride) {
        out[i] = (int) log((float) x[i] + epsilon);
    }
}

template <typename T>
void log_full_device(Tensor<T> *x, Tensor<T> *out, bool stable) {
    unsigned int size = x->get_size();
    const auto grid_dim = ceildiv(size, BLK_SIZE);

    kernel_log_full_device<<<grid_dim, BLK_SIZE>>>(
          x->get_ptr(), out->get_ptr(), size,
          (stable) ? static_cast<T>(1E-8):static_cast<T>(0));
}
template void log_full_device(Tensor<int> *x, Tensor<int> *out, bool stable);
template void log_full_device(Tensor<float> *x, Tensor<float> *out, bool stable);
template void log_full_device(Tensor<double> *x, Tensor<double> *out, bool stable);

template <typename T>
void log_full_device(cudaStream_t custream, Tensor<T> *x, Tensor<T> *out, bool stable) {
    unsigned int size = x->get_size();
    const auto grid_dim = ceildiv(size, BLK_SIZE);

    kernel_log_full_device
       <<<grid_dim, BLK_SIZE, 0, custream>>>
       (x->get_ptr(), out->get_ptr(), size,
        (stable) ? static_cast<T>(1E-8):static_cast<T>(0));
}
template void log_full_device(cudaStream_t custream, Tensor<int> *x, Tensor<int> *out, bool stable);
template void log_full_device(cudaStream_t custream, Tensor<float> *x, Tensor<float> *out, bool stable);
template void log_full_device(cudaStream_t custream, Tensor<double> *x, Tensor<double> *out, bool stable);

   
template <typename T>
__global__ void kernel_log_grad_device(T *x, T *grad, T *out, unsigned int size, T epsilon) {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < size; i += stride) {
        out[i] = grad[i] / (x[i] + epsilon);
    }
}

template <typename T>
void log_grad_device(Tensor<T> *x, Tensor<T> *grad, Tensor<T> *out, bool stable) {

   unsigned int size = x->get_size();
    const auto grid_dim = ceildiv(size, BLK_SIZE);

   kernel_log_grad_device
      <<<grid_dim, BLK_SIZE>>>
      (x->get_ptr(), grad->get_ptr(),
       out->get_ptr(), size,
       (stable) ? static_cast<T>(1E-8):static_cast<T>(0));
}
template void log_grad_device(Tensor<int> *x, Tensor<int> *grad, Tensor<int> *out, bool stable);
template void log_grad_device(Tensor<float> *x, Tensor<float> *grad, Tensor<float> *out, bool stable);
template void log_grad_device(Tensor<double> *x, Tensor<double> *grad, Tensor<double> *out, bool stable);

template <typename T>
void log_grad_device(
      cudaStream_t custream, Tensor<T> *x, Tensor<T> *grad, Tensor<T> *out,
      bool stable) {

   unsigned int size = x->get_size();
   const auto grid_dim = ceildiv(size, BLK_SIZE);

   kernel_log_grad_device
      <<<grid_dim, BLK_SIZE, 0, custream>>>
      (x->get_ptr(), grad->get_ptr(),
       out->get_ptr(), size,
       (stable) ? static_cast<T>(1E-8):static_cast<T>(0));
}
template void log_grad_device(cudaStream_t custream, Tensor<int> *x, Tensor<int> *grad, Tensor<int> *out, bool stable);
template void log_grad_device(cudaStream_t custream, Tensor<float> *x, Tensor<float> *grad, Tensor<float> *out, bool stable);
template void log_grad_device(cudaStream_t custream, Tensor<double> *x, Tensor<double> *grad, Tensor<double> *out, bool stable);
   
}  // namespace internal
}  // namespace magmadnn

#undef BLK_SIZE
