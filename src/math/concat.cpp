/**
 * @file concat.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-06-06
 * 
 * @copyright Copyright (c) 2019
 */
#include "math/concat.h"

namespace magmadnn {
namespace math {

template <typename T>
void concat(Tensor<T> *A, Tensor<T> *B, Tensor<T> *C, unsigned int axis) {
    assert(A->get_shape().size() == B->get_shape().size()); 
    assert(A->get_shape().size() == C->get_shape().size());
    unsigned int dims = A -> get_shape().size();

    // test A and B have at most 1 different dimension
    // test A and C have at most 1 different dimension
    unsigned int num_diff_B = 0;
    unsigned int num_diff_C = 0;
    unsigned int diff_index_B = axis;
    unsigned int diff_index_C = axis;
    for (unsigned int i = 0; i < dims; i ++) {
        if (A->get_shape(i) != B->get_shape(i)) {
            num_diff_B ++;
            diff_index_B = i;
        }
        if (A->get_shape(i) != C->get_shape(i)) {
            num_diff_C ++;
            diff_index_C = i;
        }
    }
    assert(num_diff_B <= 1);
    assert(diff_index_B == axis);
    assert(num_diff_C == 1);
    assert(diff_index_C == axis);
    assert(C->get_shape(axis) == A->get_shape(axis) + B->get_shape(axis));

    
    // actual concatenation
    std::vector<unsigned int> target_shape(dims, 0);
    std::vector<unsigned int> target_shape_copy(dims, 0);
    int curr_pos = target_shape.size() - 1;
    while (curr_pos >= 0) {
        curr_pos = target_shape.size() - 1;
        if (target_shape[axis] < A->get_shape(axis)) {
            C->set(target_shape, A->get(target_shape));
        } else {
            target_shape_copy = target_shape;
            target_shape_copy[axis] -= A->get_shape(axis);
            C->set(target_shape, B->get(target_shape_copy));
        }
        target_shape[curr_pos] ++;
        while(target_shape[curr_pos] == C->get_shape(curr_pos)) {
            target_shape[curr_pos] = 0;
            curr_pos --;
            if (curr_pos < 0) break;
            target_shape[curr_pos] ++;
        }
    }
}

template void concat(Tensor<int> *A, Tensor<int> *B, Tensor<int> *C, unsigned int axis);
template void concat(Tensor<float> *A, Tensor<float> *B, Tensor<float> *C, unsigned int axis);
template void concat(Tensor<double> *A, Tensor<double> *B, Tensor<double> *C, unsigned int axis);

}
}