/**
 * @file fill_internal_device.cu
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-12
 *
 * @copyright Copyright (c) 2019
 */

// #include "tensor/fill_internal.h"
#include "memory/memorymanager.h"

#define BLK_SIZE 1024

namespace magmadnn {
namespace internal {

template <typename T>
__global__ void kernel_fill_constant(T *arr, unsigned int size, T val) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < size; i += stride) {
        arr[i] = val;
    }
}

template <typename T>
void fill_constant_device(MemoryManager<T> &m, T val) {
    unsigned int size = m.get_size();
    kernel_fill_constant<<<(size + BLK_SIZE - 1) / BLK_SIZE, BLK_SIZE>>>(m.get_device_ptr(), size, val);
}
template void fill_constant_device(MemoryManager<int> &m, int val);
template void fill_constant_device(MemoryManager<float> &m, float val);
template void fill_constant_device(MemoryManager<double> &m, double val);

}  // namespace internal
}  // namespace magmadnn

#undef BLK_SIZE
