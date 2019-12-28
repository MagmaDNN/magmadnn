#include "compute/reducesum/reducesum_internal.h"

#include <cassert>

#include "math/tile.h"
#include "tensor/tensor.h"

namespace magmadnn {
namespace internal {

template <typename T>
void reduce_sum_grad(Tensor<T> *grad, int axis, Tensor<T> *out) {
    /* the gradient of reduce sum, should repeat grad along the axis that was reduced.
        Let a0, ... , an  be the shape of reduce_sum's input (also 'out'). Then
        grad should be of shape a0, ... , 1, ... an, where 1 is at index='axis'. */
    const std::vector<unsigned int> &grad_shape = grad->get_shape();
    const std::vector<unsigned int> &out_shape = out->get_shape();
    unsigned int n_grad_axes = grad_shape.size();
    unsigned int n_out_axes = out_shape.size();
    unsigned int out_size = out->get_size();

    assert(n_grad_axes == n_out_axes - 1 || n_grad_axes == n_out_axes);
    assert(T_IS_SAME_MEMORY_TYPE(grad, out));

    if (out->get_memory_type() == HOST) {
        T *grad_ptr = grad->get_ptr();
        T *out_ptr = out->get_ptr();

        /* TODO -- modularize this into one single algorithm */
        if (n_grad_axes == 1 && grad->get_size() == 1) {
            /* grad is a scalar -- fill out with the value */

            for (unsigned int i = 0; i < out_size; i++) {
                out_ptr[i] = grad_ptr[0];
            }

        } else if (n_grad_axes == 2 && grad_shape[0] == 1) {
            /* grad gets repeated along each row of out */

            unsigned int n_rows = out_shape[0];
            unsigned int n_cols = out_shape[1];

            assert(grad_shape[1] == n_cols);

            for (unsigned int i = 0; i < n_rows; i++) {
                for (unsigned int j = 0; j < n_cols; j++) {
                    out_ptr[i * n_cols + j] = grad_ptr[j]; /* out[i,j] = grad[:,j] */
                }
            }

        } else if (n_grad_axes == 2 && grad_shape[1] == 1) {
            /* grad gets repeated along each column of out */

            unsigned int n_rows = out_shape[0];
            unsigned int n_cols = out_shape[1];

            assert(grad_shape[0] == n_rows);

            for (unsigned int i = 0; i < n_rows; i++) {
                for (unsigned int j = 0; j < n_cols; j++) {
                    out_ptr[i * n_cols + j] = grad_ptr[i];
                }
            }

        } else {
            /* use math::tile for the more general case */
            math::tile(grad, out, out->get_shape(axis), axis);
        }
    }
#if defined(_HAS_CUDA_)
    else {
        reduce_sum_grad_device(grad, axis, out);
    }
#endif
}
template void reduce_sum_grad(Tensor<int> *grad, int axis, Tensor<int> *out);
template void reduce_sum_grad(Tensor<float> *grad, int axis, Tensor<float> *out);
template void reduce_sum_grad(Tensor<double> *grad, int axis, Tensor<double> *out);

}  // namespace internal
}  // namespace magmadnn
