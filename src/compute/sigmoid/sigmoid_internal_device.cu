/**
 * @file sigmoid_internal_device.cu
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-02-23
 * 
 * @copyright Copyright (c) 2019
 */
#include "compute/sigmoid/sigmoid_internal.h"

#define BLK_SIZE 1024

namespace magmadnn {
namespace internal {

template <typename T>
__global__ void kernel_fast_sigmoid_full_device(unsigned int size, T *x, T *out) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;

	for (unsigned int i = idx; i < size; i += stride) {
        out[i] = x[i] / (1 + abs(x[i]));
	}
}

template <typename T>
__global__ void kernel_sigmoid_full_device(unsigned int size, T *x, T *out) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;

	for (unsigned int i = idx; i < size; i += stride) {
        out[i] = 1 / (1 + exp(-x[i]));
	}
}


/* exp(INT_TYPE) is not defined in CUDA, so just use 1/(1+|x|) for int.
   Everything will be zero anyways. TODO: decide what to do with int sigmoid. */
template <>
__global__ void kernel_sigmoid_full_device(unsigned int size, int *x, int *out) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;

	for (unsigned int i = idx; i < size; i += stride) {
        out[i] = 1 / (1 + abs(x[i]));
	}
}


template <typename T>
void sigmoid_full_device(Tensor<T> *x, Tensor<T> *out, bool fast) {
	unsigned int size = out->get_size();

    if (fast)
        kernel_fast_sigmoid_full_device <<<(size+BLK_SIZE-1)/BLK_SIZE, BLK_SIZE>>> (size, x->get_ptr(), out->get_ptr());
    else
        kernel_sigmoid_full_device <<<(size+BLK_SIZE-1)/BLK_SIZE, BLK_SIZE>>> (size, x->get_ptr(), out->get_ptr());
}

template<> void sigmoid_full_device(Tensor<int> *x, Tensor<int> *out, bool fast) {
	/* sigmoid doesn't make much sense on integer precision */
	for (unsigned int i = 0; i < x->get_size(); i++)
		x->set(i, (int) exp(x->get(i)));
}

template void sigmoid_full_device(Tensor<float> *x, Tensor<float> *out, bool fast);
template void sigmoid_full_device(Tensor<double> *x, Tensor<double> *out, bool fast);





template <typename T>
__global__ void kernel_sigmoid_grad_device(T *output, T *grad, T *out, unsigned int size, bool is_grad_scalar) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;

	for (unsigned int i = idx; i < size; i += stride) {
		out[i] = grad[(is_grad_scalar) ? 0 : i] * output[i] * (1 - output[i]);
	}
}

template <typename T>
void sigmoid_grad_device(Tensor<T> *output, Tensor<T> *grad, Tensor<T> *out) {
	unsigned int size = out->get_size();

	kernel_sigmoid_grad_device <<< (size+BLK_SIZE-1)/BLK_SIZE, BLK_SIZE >>> (output->get_ptr(), grad->get_ptr(), out->get_ptr(), size, (grad->get_size() == 1));
}
template void sigmoid_grad_device(Tensor<int> *output, Tensor<int> *grad, Tensor<int> *out);
template void sigmoid_grad_device(Tensor<float> *output, Tensor<float> *grad, Tensor<float> *out);
template void sigmoid_grad_device(Tensor<double> *output, Tensor<double> *grad, Tensor<double> *out);


}   // namespace internal
}   // namespace magmadnn


#undef BLK_SIZE