/**
 * @file tensor_utilities.h
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-07-24
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include <vector>

#include "data_types.h"
#include "tensor/tensor.h"
#include "types.h"

namespace magmadnn {
namespace utilities {

inline bool do_tensors_match(DataType dtype, memory_t mem_type,
                             const std::initializer_list<std::reference_wrapper<const Tensor>>& tensors) {
    for (const auto& t : tensors) {
        if ((t.get().dtype() != dtype) || (t.get().get_memory_type() != mem_type)) return false;
    }
    return true;
}

inline bool do_tensors_match(DataType dtype, memory_t mem_type,
                             const std::vector<std::reference_wrapper<const Tensor>>& tensors) {
    for (const auto& t : tensors) {
        if ((t.get().dtype() != dtype) || (t.get().get_memory_type() != mem_type)) return false;
    }
    return true;
}

}  // namespace utilities
}  // namespace magmadnn