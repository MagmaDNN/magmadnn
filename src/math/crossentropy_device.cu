/**
 * @file crossentropy.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-06-12
 *
 * @copyright Copyright (c) 2019
 *
 */
#include "math/crossentropy.h"

#define BLK_SIZE 256

namespace magmadnn {
namespace math {

/* Define atomicAdd(double*, double) for architectures <600.
    This code is from the Cuda Developer Documentation
    @ https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
*/
#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double *address, double val) {
    unsigned long long int *address_as_ull = (unsigned long long int *) address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

template <typename T>
__device__ T log_device(T val) {
    return log(val);
}
template <>
__device__ int log_device(int val) {
    return (int) log((float) val);
}

template <typename T>
__global__ void kernel_crossentropy_device(T *predicted, T *ground_truth, T *_partial_out, unsigned int n_samples,
                                           unsigned int n_classes) {
    __shared__ T cache[BLK_SIZE];
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;
    unsigned int cacheIdx = threadIdx.x;
    int i;

    T tmp_sum = (T) 0;
    for (unsigned int i = idx; i < n_samples * n_classes; i += stride) {
        if (predicted[i] <= 0) continue;
        tmp_sum += log_device(predicted[i]) * ground_truth[i];
    }

    cache[cacheIdx] = tmp_sum;

    __syncthreads();

    i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIdx < i) {
            cache[cacheIdx] += cache[cacheIdx + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (cacheIdx == 0) {
        _partial_out[blockIdx.x] = cache[0];
    }
}

template <typename T>
void crossentropy_device(Tensor<T> *predicted, Tensor<T> *ground_truth, Tensor<T> *out) {
    unsigned int n_classes = predicted->get_shape(1);
    unsigned int n_samples = predicted->get_shape(0);
    unsigned int size = n_samples * n_classes;

    T *_partial_sum, *_dev_partial_sum;
    T sum;

    /* instantiate _partial_sum and _dev_partial_sum */
    _partial_sum = new T[BLK_SIZE];
    cudaMalloc((void **) &_dev_partial_sum, BLK_SIZE * sizeof(T));

    kernel_crossentropy_device<<<(size + BLK_SIZE - 1) / BLK_SIZE, BLK_SIZE>>>(
        predicted->get_ptr(), ground_truth->get_ptr(), _dev_partial_sum, n_samples, n_classes);

    /* copy values from device to host and sum the individual block sums */
    cudaMemcpy(_partial_sum, _dev_partial_sum, BLK_SIZE * sizeof(T), cudaMemcpyDeviceToHost);
    sum = (T) 0;
    for (unsigned int i = 0; i < BLK_SIZE; i++) {
        sum += _partial_sum[i];
    }
    sum /= (T) n_samples;

    out->set(0, -sum);

    delete _partial_sum;
    cudaFree(_dev_partial_sum);
}
template void crossentropy_device(Tensor<int> *predicted, Tensor<int> *ground_truth, Tensor<int> *out);
template void crossentropy_device(Tensor<float> *predicted, Tensor<float> *ground_truth, Tensor<float> *out);
template void crossentropy_device(Tensor<double> *predicted, Tensor<double> *ground_truth, Tensor<double> *out);

}  // namespace math
}  // namespace magmadnn

#undef BLK_SIZE
