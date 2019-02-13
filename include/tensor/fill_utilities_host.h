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
#include <stdio.h>
#include "memory/memorymanager.h"

#ifdef _HAS_CUDA_
#include "fill_utilities_device.h"
#endif

namespace skepsi {

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
