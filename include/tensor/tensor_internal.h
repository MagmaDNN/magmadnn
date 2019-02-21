/**
 * @file tensor_internal.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-12
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "memory/memorymanager.h"
#include "fill_internal.h"


namespace skepsi {
namespace internal {


template <typename T>
void fill_memory(memorymanager<T> &m, tensor_filler_t<T> filler);

}   // namespace internal
}   // namespace skepsi
