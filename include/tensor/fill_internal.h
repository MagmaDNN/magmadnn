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
#include "memory/memorymanager.h"

namespace skepsi {

#ifdef _HAS_CUDA_
template <typename T>
void fill_constant_device(memorymanager<T> &m, T val);
#endif

template <typename T>
void fill_uniform(memorymanager<T> &m, const std::vector<double>& params);


template <typename T>
void fill_glorot(memorymanager<T> &m, const std::vector<double>& params);


template <typename T>
void fill_constant(memorymanager<T> &m, const std::vector<double>& params);


} // namespace skepsi
