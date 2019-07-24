/**
 * @file tile.cpp
 * @author Sedrick Keh
 * @version 0.1
 * @date 2019-06-17
 *
 * @copyright Copyright (c) 2019
 */
#include "math/tile.h"

namespace magmadnn {
namespace math {

template <typename T>
void tile(const Tensor &A, Tensor &B, index_t t, index_t axis) {
    // assert(A.shape().size() == B.shape().size());
    MAGMADNN_ASSERT(A.shape().size() == B.shape().size(), "Invalid Tensor Dimensions");
    MAGMADNN_ASSERT(::magmadnn::utilities::do_tensors_match(GetDataType<T>::value, B.get_memory_type(), {A, B}),
                    "Invalid Tensors");
    unsigned int dims = A.shape().size();

    // test A and B have at most 1 different dimension
    unsigned int num_diff_B = 0;
    unsigned int diff_index_B = axis;
    for (unsigned int i = 0; i < dims; i++) {
        if (A.shape(i) != B.shape(i)) {
            num_diff_B++;
            diff_index_B = i;
        }
    }

    assert(num_diff_B <= 1);
    assert(diff_index_B == axis);
    assert(B.shape(axis) == A.shape(axis) * t);

    // actual tiling
    std::vector<index_t> target_shape(dims, 0);
    std::vector<index_t> target_shape_copy(dims, 0);
    int curr_pos = target_shape.size() - 1;
    while (curr_pos >= 0) {
        curr_pos = target_shape.size() - 1;

        target_shape_copy = target_shape;
        target_shape_copy[axis] = 0;
        B.set<T>(target_shape, A.get<T>(target_shape_copy));

        target_shape[curr_pos]++;
        while (target_shape[curr_pos] == B.shape(curr_pos)) {
            target_shape[curr_pos] = 0;
            curr_pos--;
            if (curr_pos < 0) break;
            target_shape[curr_pos]++;
        }
    }
}
#define comp(type) template void tile<type>(const Tensor &, Tensor &, index_t, index_t);
CALL_FOR_ALL_TYPES(comp)
#undef comp

}  // namespace math
}  // namespace magmadnn