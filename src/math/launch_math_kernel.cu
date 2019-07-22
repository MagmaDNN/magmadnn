/**
 * @file launch_math_kernel.cpp
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-07-22
 *
 * @copyright Copyright (c) 2019
 */
#include "math/launch_math_kernel.h"

#define BLK_SIZE 1024

namespace magmadnn {
namespace math {


template <typename Mapper, typename... Args>
__global__ void kernel_mapped(size_t SIZE, Args... args) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < SIZE; i += stride) {
        Mapper::map(i, args);
    }
}


template <typename Mapper, typename Dtype, typename... Args>
inline static void launchMappedKernelGPU(const size_t SIZE, Args... args) {
    kernel_mapped<<<(SIZE + BLK_SIZE-1) / BLK_SIZE, BLK_SIZE>>>(SIZE, args);
}

}  // namespace math
}  // namespace magmadnn

#undef BLK_SIZE