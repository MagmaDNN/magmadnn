/**
 * @file tensor_internal.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-12
 *
 * @copyright Copyright (c) 2019
 */
#include "tensor/tensor_internal.h"

namespace magmadnn {
namespace internal {

void fill_memory(MemoryManager &m, tensor_filler_t filler) {
    FOR_ALL_DTYPES(m.dtype(), Dtype, {
        std::vector<Dtype> vals(filler.values.begin(), filler.values.end());

        switch (filler.fill_type) {
            case UNIFORM:
                fill_uniform(m, vals);
                break;
            case GLOROT:
                fill_glorot(m, vals);
                break;
            case MASK:
                fill_mask(m, vals);
                break;
            case CONSTANT:
                fill_constant(m, vals);
                break;
            case ZERO:
                fill_constant<Dtype>(m, {static_cast<Dtype>(0)});
                break;
            case ONE:
                fill_constant<Dtype>(m, {static_cast<Dtype>(1)});
                break;
            case DIAGONAL:
                fill_diagonal(m, vals);
                break;
            case IDENTITY:
                fill_diagonal<Dtype>(m, {static_cast<Dtype>(1)});
                break;
            case NONE:
                break;
        }
    });
}

}  // namespace internal
}  // namespace magmadnn
