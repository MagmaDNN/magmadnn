/**
 * @file tensor_internal.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-12
 * 
 * @copyright Copyright (c) 2019
 */
#include "tensor/tensor_internal.h"

namespace skepsi {
namespace internal {

template <typename T>
void fill_memory(MemoryManager<T> &m, tensor_filler_t<T> filler) {
    switch (filler.fill_type) {
        case UNIFORM:
            fill_uniform(m, filler.values); break;
        case GLOROT:
            fill_glorot(m, filler.values); break;
        case CONSTANT:
            fill_constant(m, filler.values); break;
        case ZERO:
            fill_constant(m, {(T)0}); break;
        case ONE:
            fill_constant(m, {(T)1}); break;
        case DIAGONAL:
            fill_diagonal(m, filler.values); break;
        case IDENTITY:
            fill_diagonal(m, {(T)1}); break;
        case NONE: break;
    }
}
template void fill_memory(MemoryManager<int>&, tensor_filler_t<int>);
template void fill_memory(MemoryManager<float>&, tensor_filler_t<float>);
template void fill_memory(MemoryManager<double>&, tensor_filler_t<double>);

} // namespace internal
} // namespace skepsi
