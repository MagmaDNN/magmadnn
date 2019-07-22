/**
 * @file launch_math_kernel.h
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-07-22
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "tensor/tensor.h"
#include "types.h"

namespace magmadnn {
namespace math {

template <typename Mapper, typename Dtype, typename... Args>
inline static void launchMappedKernelCPU(const size_t SIZE, Args... args) {
#if defined(_USE_OPENMP_)
    const int n_threads = 1; /* TODO -- calculate num threads */
    if (n_threads < 2) {
        for (size_t i = 0; i < SIZE; i++) {
            Mapper::map(i, args);
        }
    } else {
#pragma omp parallel for num_threads(n_threads)
        for (size_t i = 0; i < SIZE; i++) {
            Mapper::map(i, args);
        }
    }
#else
    for (size_t i = 0; i < SIZE; i++) {
        Mapper::map(i, args);
    }
#endif
}


template <typename Mapper, typename Dtype, typename... Args>
inline static void launchMappedKernelGPU(const size_t SIZE, Args... args);

}  // namespace math
}  // namespace magmadnn