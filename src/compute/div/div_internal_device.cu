
#include "compute/div/div_internal.h"

namespace magmadnn {
namespace internal {
 
template <typename T>
__global__ void tensor_div_tensor_full_device(T *a, T *b, T *out, unsigned int size) {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < size; i += stride) {
        if (b[i] == (T) 0) continue;
        out[i] = a[i] / b[i];
    }
}
template <typename T>
void tensor_div_tensor_full_device(Tensor<T> *a, Tensor<T> *b, Tensor<T> *out) {
    unsigned int size = out->get_size();
    tensor_div_tensor_full_device <<< 1, size >>> ()
}
template void tensor_div_tensor_full_device(Tensor<int> *a, Tensor<int> *b, Tensor<int> *out);
template void tensor_div_tensor_full_device(Tensor<float> *a, Tensor<float> *b, Tensor<float> *out);
template void tensor_div_tensor_full_device(Tensor<double> *a, Tensor<double> *b, Tensor<double> *out);
 
 
template <typename T>
__global__ void kernel_tensor_div_scalar_full_device(T *a, T scalar, T *out, unsigned int size) {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < size; i += stride) {
        out[i] = a[i] / scalar;
    }
}
template <typename T>
void tensor_div_scalar_full_device(Tensor<T> *a, T scalar, Tensor<T> *out) {
    if (scalar == (T) 0) return;
    unsigned int size = out->get_size();
    kernel_tensor_div_scalar_full_device <<< 1, size >>> (a->get_ptr(), scalar, out->get_ptr(), size);
}
template void tensor_div_scalar_full_device(Tensor<int> *a, int scalar, Tensor<int> *out);
template void tensor_div_scalar_full_device(Tensor<float> *a, float scalar, Tensor<float> *out);
template void tensor_div_scalar_full_device(Tensor<double> *a, float scalar, Tensor<double> *out);
 
 
template <typename T>
__global__ void kernel_scalar_div_tensor_full_device(T scalar, T *a, T *out, unsigned int size) {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < size; i += stride) {
        if (a[i] == (T) 0) continue;
        out[i] = scalar / a[i];
    }
}
template <typename T>
void scalar_div_tensor_full_device(T scalar, Tensor<T> *a, Tensor<T> *out) {
    unsigned int size = out->get_size();
    kernel_scalar_div_tensor_full_device <<< 1, size >>> (scalar, a->get_ptr(), out->get_ptr(), size);    
}
template void tensor_div_scalar_full_device(int scalar, Tensor<int> *b, Tensor<int> *out);
template void tensor_div_scalar_full_device(float scalar, Tensor<float> *b, Tensor<float> *out);
template void tensor_div_scalar_full_device(double scalar, Tensor<double> *b, Tensor<double> *out);
 
}   // namespace op
}   // namespace magmadnn
