/**
 * @file tanh_internal_device.cu
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-02-23
 * 
 * @copyright Copyright (c) 2019
 */
#include "compute/tanh/tanh_internal.h"

#define BLK_SIZE 1024

namespace magmadnn {
namespace internal {


template <typename T>
__global__ void kernel_tanh_full_device(unsigned int size, T *x, T *out) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;

	for (unsigned int i = idx; i < size; i += stride) {
        out[i] = tanh(x[i]);
	}
}

/* tanh(INT_TYPE) is not defined in CUDA. TODO: determine what to do for 
   int types with tanh */
template <>
__global__ void kernel_tanh_full_device(unsigned int size, int *x, int *out) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;

	/* tanh : R -> (-1,1)  which is 0 in the integers */
	for (unsigned int i = idx; i < size; i += stride) {
        out[i] = 0;
	}
}

template <typename T>
void tanh_full_device(Tensor<T> *x, Tensor<T> *out) {
	unsigned int size = x->get_size();
    kernel_tanh_full_device <<<(size+BLK_SIZE-1)/BLK_SIZE,BLK_SIZE>>> (size, x->get_ptr(), out->get_ptr());
}

template<> void tanh_full_device(Tensor<int> *x, Tensor<int> *out) {
	for (unsigned int i = 0; i < x->get_size(); i++)
		out->set(i, (int)tanh(x->get(i)));
}

template void tanh_full_device(Tensor<float> *x, Tensor<float> *out);
template void tanh_full_device(Tensor<double> *x, Tensor<double> *out);

}   // namespace internal
}   // namespace magmadnn

#undef BLK_SIZE
