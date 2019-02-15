/**
 * @file tensor_utilities.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-12
 * 
 * @copyright Copyright (c) 2019
 */
#include "tensor/tensor_utilities.h"

namespace skepsi {

template <typename T>
void fill_memory(memorymanager<T> &m, tensor_filler_t filler) {
    switch (filler.fill_type) {
        case UNIFORM:
            fill_uniform(m, filler.values); break;
        case GLOROT:
            fill_glorot(m, filler.values); break;
        case CONSTANT:
            fill_constant(m, filler.values); break;
        case ZERO:
            fill_constant(m, {0.0}); break;
        case ONE:
            fill_constant(m, {1.0}); break;
        case NONE: break;
    }
}
template void fill_memory(memorymanager<int>&, tensor_filler_t);
template void fill_memory(memorymanager<float>&, tensor_filler_t);
template void fill_memory(memorymanager<double>&, tensor_filler_t);

} // namespace skepsi
