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
__device__ __forceinline__ T log_device(T val) {
    return log(val);
}
#define SPECIALIZE_LOGDEVICE_INT(int_type) template <> __device__ __forceinline__ int_type log_device(int_type val) { return (int_type) log((float)val); }
CALL_FOR_ALL_INT_TYPES(SPECIALIZE_LOGDEVICE_INT)
#undef SPECIALIZE_LOGDEVICE_INT

template <typename T>
__global__ void kernel_crossentropy_device(const T *predicted, const T *ground_truth, T *_partial_out, unsigned int n_samples,
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
void crossentropy_device(const Tensor& predicted, const Tensor& ground_truth, Tensor &out) {
    index_t n_classes = predicted.shape(1);
    index_t n_samples = predicted.shape(0);
    size_t size = n_samples * n_classes;

    T *_partial_sum, *_dev_partial_sum;
    T sum;

    /* instantiate _partial_sum and _dev_partial_sum */
    _partial_sum = new T[BLK_SIZE];
    cudaMalloc((void **) &_dev_partial_sum, BLK_SIZE * sizeof(T));

    kernel_crossentropy_device<<<(size + BLK_SIZE - 1) / BLK_SIZE, BLK_SIZE>>>(
        predicted.get_ptr<T>(), ground_truth.get_ptr<T>(), _dev_partial_sum, n_samples, n_classes);

    /* copy values from device to host and sum the individual block sums */
    cudaMemcpy(_partial_sum, _dev_partial_sum, BLK_SIZE * sizeof(T), cudaMemcpyDeviceToHost);
    sum = (T) 0;
    
    for (unsigned int i = 0; i < BLK_SIZE; i++) {
        sum += _partial_sum[i];
    }
    sum /= (T) n_samples;

    out.set<T>(0, -sum);

    delete _partial_sum;
    cudaFree(_dev_partial_sum);
}
#define COMPILE_CROSSENTROPY_DEVICE(type) template void crossentropy_device<type>(const Tensor&, const Tensor&, Tensor&);
CALL_FOR_ALL_TYPES(COMPILE_CROSSENTROPY_DEVICE)
#undef COMPILE_CROSSENTROPY_DEVICE

}  // namespace math
}  // namespace magmadnn

#undef BLK_SIZE
