/**
 * @file fill_internal.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-12
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include <random>
#include <vector>

#include "magmadnn/utilities_internal.h"
#include "memory/memorymanager.h"

#if defined(MAGMADNN_HAVE_CUDA)
#include <cuda.h>
#include <curand.h>
#endif

namespace magmadnn {
namespace internal {

#if defined(MAGMADNN_HAVE_CUDA)
/** Calls cuda kernel to fill the memorymanager with constant val.
 * @tparam T
 * @param m
 * @param val
 */
template <typename T>
void fill_constant_device(MemoryManager<T> &m, T val);

template <typename T>
void fill_constant_device(cudaStream_t custream, MemoryManager<T> &m, T val);
#endif

/** Fills the memory manager with a uniform distribution.
 * @tparam T
 * @param m memorymanager that will be filled
 * @param params
 */
template <typename T>
void fill_uniform(MemoryManager<T> &m, const std::vector<T> &params);

/** Fills the memorymanager with a modified normal distribution (per Glorot et. al.).
 * @tparam T
 * @param m memorymanager to be filled.
 * @param params
 */
template <typename T>
void fill_glorot(MemoryManager<T> &m, const std::vector<T> &params);

/** Fills the memorymanager zeroes and ones using a Bernoulli distribution.
 * @tparam T
 * @param m memorymanager to be filled.
 * @param params
 */
template <typename T>
void fill_mask(MemoryManager<T> &m, const std::vector<T> &params);

/** Fills the memorymanager's diagonal elements. Assumes the memory manager is square.
 * If one value is given for params, then it is applied to all diagonals. Otherwise the list
 * is applied to fill the diagonals.
 * @tparam T
 * @param m
 * @param params
 */
template <typename T>
void fill_diagonal(MemoryManager<T> &m, const std::vector<T> &params);

/** Fills the memorymanager with the same constant value.
 * @tparam T
 * @param m memorymanager to be filled.
 * @param params
 */
template <typename T>
void fill_constant(MemoryManager<T> &m, const std::vector<T> &params);

}  // namespace internal
}  // namespace magmadnn
