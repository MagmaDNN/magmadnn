/**
 * @file fill_internal_device.cu
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-12
 *
 * @copyright Copyright (c) 2019
 */

#include "magmadnn/math.h"
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
    const auto grid_dim = ceildiv(size, BLK_SIZE);

    kernel_fill_constant<<<grid_dim, BLK_SIZE>>>(m.get_device_ptr(), size, val);
}
template void fill_constant_device(MemoryManager<int> &m, int val);
template void fill_constant_device(MemoryManager<float> &m, float val);
template void fill_constant_device(MemoryManager<double> &m, double val);

template <typename T>
void fill_constant_device(cudaStream_t custream, MemoryManager<T> &m, T val) {
    unsigned int size = m.get_size();
    const auto grid_dim = ceildiv(size, BLK_SIZE);

    kernel_fill_constant
       <<<grid_dim, BLK_SIZE, 0, custream>>>
       (m.get_device_ptr(), size, val);
}
template void fill_constant_device(
      cudaStream_t custream, MemoryManager<int> &m, int val);
template void fill_constant_device(
      cudaStream_t custream, MemoryManager<float> &m, float val);
template void fill_constant_device(
      cudaStream_t custream, MemoryManager<double> &m, double val);
   
}  // namespace internal
}  // namespace magmadnn

#undef BLK_SIZE
