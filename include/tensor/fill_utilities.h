/**
 * @file fill_utilities_host.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-12
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once

#include <vector>
#include <string.h>
#include "memory/memorymanager.h"

namespace skepsi {

#ifdef _HAS_CUDA_
template <typename T>
__global__ void kernel_fill_glorot(T *arr, double *vals);

template <typename T>
__global__ void kernel_fill_uniform(T *arr, double *vals);
#endif

template <typename T>
void fill_uniform(memorymanager<T> &m, const std::vector<double>& params);


template <typename T>
void fill_glorot(memorymanager<T> &m, const std::vector<double>& params);


template <typename T>
void fill_constant(memorymanager<T> &m, const std::vector<double>& params);


template <typename T>
void fill_zero(memorymanager<T> &m, const std::vector<double>& params);


template <typename T>
void fill_one(memorymanager<T> &m, const std::vector<double>& params);

} // namespace skepsi
