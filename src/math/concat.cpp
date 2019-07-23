/**
 * @file concat.cpp
 * @author Sedrick Keh
 * @version 0.1
 * @date 2019-06-14
 *
 * @copyright Copyright (c) 2019
 */
#include "math/concat.h"

namespace magmadnn {
namespace math {

template <typename T>
void concat(const Tensor &A, const Tensor &B, Tensor &C, index_t axis) {
    // assert(A->get_shape().size() == B->get_shape().size());
    // assert(A->get_shape().size() == C->get_shape().size());
    MAGMADNN_ASSERT(A.shape().size() == B.shape().size(), "invalid shapes");
    MAGMADNN_ASSERT(A.shape().size() == C.shape().size(), "invalid shapes");

    index_t dims = A.shape().size();

    // test A and B have at most 1 different dimension
    // test A and C have at most 1 different dimension
    unsigned int num_diff_B = 0;
    unsigned int num_diff_C = 0;
    index_t diff_index_B = axis;
    index_t diff_index_C = axis;

    for (unsigned int i = 0; i < dims; i++) {
        if (A.shape(i) != B.shape(i)) {
            num_diff_B++;
            diff_index_B = i;
        }
        if (A.shape(i) != C.shape(i)) {
            num_diff_C++;
            diff_index_C = i;
        }
    }

    MAGMADNN_ASSERT(num_diff_B <= 1, "too many axis variations");
    MAGMADNN_ASSERT(diff_index_B == axis, "invalid axis");
    MAGMADNN_ASSERT(num_diff_C == 1, "too many axis variations");
    MAGMADNN_ASSERT(diff_index_C == axis, "invalid axis");
    MAGMADNN_ASSERT(C.shape(axis) == A.shape(axis) + B.shape(axis), "invalid shapes");

    // actual concatenation
    std::vector<index_t> target_shape(dims, 0);
    std::vector<index_t> target_shape_copy(dims, 0);
    int curr_pos = target_shape.size() - 1;
    while (curr_pos >= 0) {
        curr_pos = target_shape.size() - 1;
        if (target_shape[axis] < A.shape(axis)) {
            C.set<T>(target_shape, A.get<T>(target_shape));
        } else {
            target_shape_copy = target_shape;
            target_shape_copy[axis] -= A.shape(axis);
            C.set<T>(target_shape, B.get<T>(target_shape_copy));
        }
        target_shape[curr_pos]++;
        while (target_shape[curr_pos] == C.shape(curr_pos)) {
            target_shape[curr_pos] = 0;
            curr_pos--;
            if (curr_pos < 0) break;
            target_shape[curr_pos]++;
        }
    }
}
#define COMPILE_CONCAT(type) template void concat<type>(const Tensor &, const Tensor &, Tensor &, index_t);
CALL_FOR_ALL_TYPES(COMPILE_CONCAT)
#undef COMPILE_CONCAT

}  // namespace math
}  // namespace magmadnn