/**
 * @file fill_internal.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-12
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once

#include <vector>
#include <random>
#include "memory/memorymanager.h"

#if defined(_HAS_CUDA_)
#include <cuda.h>
#endif

namespace skepsi {
namespace internal {

#if defined(_HAS_CUDA_)
/** Calls cuda kernel to fill the memorymanager with constant val.
 * @tparam T 
 * @param m 
 * @param val 
 */
template <typename T>
void fill_constant_device(memorymanager<T> &m, T val);
#endif

/** Fills the memory manager with a uniform distribution.
 * @tparam T 
 * @param m memorymanager that will be filled
 * @param params 
 */
template <typename T>
void fill_uniform(memorymanager<T> &m, const std::vector<T>& params);

/** Fills the memorymanager with a modified normal distribution (per Glorot et. al.).
 * @tparam T 
 * @param m memorymanager to be filled.
 * @param params 
 */
template <typename T>
void fill_glorot(memorymanager<T> &m, const std::vector<T>& params);

/** Fills the memorymanager with the same constant value.
 * @tparam T 
 * @param m memorymanager to be filled.
 * @param params 
 */
template <typename T>
void fill_constant(memorymanager<T> &m, const std::vector<T>& params);

} // namespace internal
} // namespace skepsi
