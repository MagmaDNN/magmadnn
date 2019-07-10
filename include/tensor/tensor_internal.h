/**
 * @file tensor_internal.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-12
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "fill_internal.h"
#include "memory/memorymanager.h"

namespace magmadnn {
namespace internal {

/** Uses the filler to fill memorymanager m. Works independent of memory type.
 * @tparam T
 * @param m memorymanager to be filled
 * @param filler how to fill m
 */
template <typename T>
void fill_memory(MemoryManager<T> &m, tensor_filler_t<T> filler);

}  // namespace internal
}  // namespace magmadnn
