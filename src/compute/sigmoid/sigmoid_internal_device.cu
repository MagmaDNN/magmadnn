/**
 * @file sigmoid_internal_device.cu
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-23
 * 
 * @copyright Copyright (c) 2019
 */
 #include "compute/sigmoid/sigmoid_internal.h"

 namespace skepsi {
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

template <typename T>
void sigmoid_full_device(tensor<T> *x, bool fast) {
    if (fast)
        kernel_fast_sigmoid_full_device <<<x->get_size(), 1>>> (x->get_size(), x->get_ptr());
    else
        kernel_sigmoid_full_device <<<x->get_size(), 1>>> (x->get_size(), x->get_ptr());
}
template void sigmoid_full_device(tensor<int> *x, bool fast);
template void sigmoid_full_device(tensor<float> *x, bool fast);
template void sigmoid_full_device(tensor<double> *x, bool fast);

 }   // namespace internal
 }   // namespace skepsi