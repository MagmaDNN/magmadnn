/**
 * @file sigmoid_internal_device.cu
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-23
 * 
 * @copyright Copyright (c) 2019
 */
 #include "compute/sigmoid/sigmoid_internal.h"

namespace magmadnn {
namespace internal {

template <typename T>
__global__ void kernel_fast_sigmoid_full_device(unsigned int size, T *x) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;

	for (unsigned int i = idx; i < size; i += stride) {
        x[i] = x[i] / (1 + abs(x[i]));
	}
}

template <typename T>
__global__ void kernel_sigmoid_full_device(unsigned int size, T *x) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;

	for (unsigned int i = idx; i < size; i += stride) {
        x[i] = 1 / (1 + exp(-x[i]));
	}
}


/* exp(INT_TYPE) is not defined in CUDA, so just use 1/(1+|x|) for int.
   Everything will be zero anyways. TODO: decide what to do with int sigmoid. */
template <>
__global__ void kernel_sigmoid_full_device(unsigned int size, int *x) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;

	for (unsigned int i = idx; i < size; i += stride) {
        x[i] = 1 / (1 + abs(x[i]));
	}
}


template <typename T>
void sigmoid_full_device(Tensor<T> *x, bool fast) {
    if (fast)
        kernel_fast_sigmoid_full_device <<<x->get_size(), 1>>> (x->get_size(), x->get_ptr());
    else
        kernel_sigmoid_full_device <<<x->get_size(), 1>>> (x->get_size(), x->get_ptr());
}

template<> void sigmoid_full_device(Tensor<int> *x, bool fast) {
	/* sigmoid doesn't make much sense on integer precision */
	for (unsigned int i = 0; i < x->get_size(); i++)
		x->set(i, (int) exp(x->get(i)));
}

template void sigmoid_full_device(Tensor<float> *x, bool fast);
template void sigmoid_full_device(Tensor<double> *x, bool fast);

}   // namespace internal
}   // namespace magmadnn
