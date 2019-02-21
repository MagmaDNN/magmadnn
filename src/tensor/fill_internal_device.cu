/**
 * @file fill_internal_device.cu
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-12
 * 
 * @copyright Copyright (c) 2019
 */

#include "tensor/fill_internal.h"

#ifdef _HAS_CUDA_

namespace skepsi {
namespace internal {

template <typename T>
__global__ void kernel_fill_constant(T* arr, unsigned int size, T val) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;

	for (unsigned int i = idx; i < size; i += stride) {
		arr[i] = val;
	}
}

template <typename T>
void fill_constant_device(memorymanager<T> &m, T val) {
	kernel_fill_constant <<<m.get_size(), 1>>> (m.get_device_ptr(), m.get_size(), val);
}
template void fill_constant_device(memorymanager<int> &m, int val);
template void fill_constant_device(memorymanager<float> &m, float val);
template void fill_constant_device(memorymanager<double> &m, double val);

} // namespace internal
} // namespace skepsi
#endif
