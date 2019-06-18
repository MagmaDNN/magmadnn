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
void tile(Tensor<T> *A, Tensor<T> *B, unsigned int t, unsigned int axis) {
    assert(A->get_shape().size() == B->get_shape().size()); 
    unsigned int dims = A -> get_shape().size();

    // test A and B have at most 1 different dimension
    unsigned int num_diff_B = 0;
    unsigned int diff_index_B = axis;
    for (unsigned int i = 0; i < dims; i ++) {
        if (A->get_shape(i) != B->get_shape(i)) {
            num_diff_B ++;
            diff_index_B = i;
        }
    }

    assert(num_diff_B <= 1);
    assert(diff_index_B == axis);
    assert(B->get_shape(axis) == A->get_shape(axis) * t);
    
    // actual tiling
    std::vector<unsigned int> target_shape(dims, 0);
    std::vector<unsigned int> target_shape_copy(dims, 0);
    int curr_pos = target_shape.size() - 1;
    while (curr_pos >= 0) {
        curr_pos = target_shape.size() - 1;
        
        target_shape_copy = target_shape;
        target_shape_copy[axis] = 0;
        B->set(target_shape, A->get(target_shape_copy));

        target_shape[curr_pos] ++;
        while(target_shape[curr_pos] == B->get_shape(curr_pos)) {
            target_shape[curr_pos] = 0;
            curr_pos --;
            if (curr_pos < 0) break;
            target_shape[curr_pos] ++;
        }
    }
}

template void tile(Tensor<int> *A, Tensor<int> *B, unsigned int t, unsigned int axis);
template void tile(Tensor<float> *A, Tensor<float> *B, unsigned int t, unsigned int axis);
template void tile(Tensor<double> *A, Tensor<double> *B, unsigned int t, unsigned int axis);

}
}