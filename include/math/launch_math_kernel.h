/**
 * @file launch_math_kernel.h
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-07-22
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "data_types.h"
#include "math/binary_math_operations.h"
#include "mdnn_device_types.h"
#include "tensor/tensor.h"
#include "types.h"

namespace magmadnn {
namespace math {

template <DeviceType dev, typename Mapper>
struct ParallelLauncher {
   public:
    template <typename... Args>
    inline static void launchMappedKernel(const size_t SIZE, Args... args) {
        LOG(ERROR) << "ParallelLauncher not defined for this device type.\n";
    }
};

template <typename Mapper>
struct ParallelLauncher<CPU, Mapper> {
   public:
    template <typename... Args>
    inline static void launchMappedKernel(const size_t SIZE, Args... args) {
#if defined(_USE_OPENMP_)
        const int n_threads = 1; /* TODO -- calculate num threads */
        if (n_threads < 2) {
            for (size_t i = 0; i < SIZE; i++) {
                Mapper::map(i, args...);
            }
        } else {
#pragma omp parallel for num_threads(n_threads)
            for (size_t i = 0; i < SIZE; i++) {
                Mapper::map(i, args...);
            }
        }
#else
        for (size_t i = 0; i < SIZE; i++) {
            Mapper::map(i, args...);
        }
#endif
    }
};

/* We only want to include these if useing gpu and cudacc is set (i.e. nvcc is compiling) */
#if defined(_HAS_CUDA_) && defined(__CUDACC__)

#define BLK_SIZE 1024

template <typename Mapper, typename... Args>
__global__ void kernel_mapped(size_t SIZE, Args... args) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < SIZE; i += stride) {
        Mapper::map(i, args...);
    }
}

template <typename Mapper>
struct ParallelLauncher<GPU, Mapper> {
   public:
    template <typename... Args>
    inline static void launchMappedKernel(const size_t SIZE, Args... args) {
        kernel_mapped<Mapper><<<(SIZE + BLK_SIZE - 1) / BLK_SIZE, BLK_SIZE>>>(SIZE, args...);
    }
};

#undef BLK_SIZE
#endif

}  // namespace math
}  // namespace magmadnn