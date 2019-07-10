/**
 * @file dotop.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-22
 *
 * @copyright Copyright (c) 2019
 */
#include "compute/dot/dotop.h"

namespace magmadnn {
namespace op {

template <typename T>
Operation<T> *dot(Operation<T> *a, Operation<T> *b, bool copy, bool needs_grad) {
    std::vector<unsigned int> const &a_shape = a->get_output_shape();
    std::vector<unsigned int> const &b_shape = b->get_output_shape();
    unsigned int a_axes, b_axes;

    a_axes = a_shape.size();
    b_axes = b_shape.size();

    if (a_axes == 1 && b_axes == 1) {
        /* vector-vector: inner product */
        std::fprintf(stderr, "inner product not yet defined.\n");
    } else if (a_axes == 1 && a_shape[0] == 1) {
        /* scalar-tensor: scalar product */
        return scalarproduct<T>(a, b, copy, needs_grad);
    } else if (b_axes == 1 && b_shape[0] == 1) {
        /* scalar-tensor: scalar product */
        return scalarproduct(b, a, copy, needs_grad);
    } else if (a_axes == 2 && b_axes == 2) {
        /* matrix-matrix: matmul */
        return matmul(a, b, needs_grad);
    } else {
        /* tensor-tensor: sum over product of last row of a and second-to-last of b */
        std::fprintf(stderr, "tensor-tensor product not yet defined.\n");
    }

    return NULL;
}
template Operation<int> *dot(Operation<int> *a, Operation<int> *b, bool copy, bool needs_grad);
template Operation<float> *dot(Operation<float> *a, Operation<float> *b, bool copy, bool needs_grad);
template Operation<double> *dot(Operation<double> *a, Operation<double> *b, bool copy, bool needs_grad);

}  // namespace op
}  // namespace magmadnn