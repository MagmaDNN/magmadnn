
#include "compute/reducesum/reducesum_internal.h"

#define BLK_SIZE 1024
#define BLK_2D_SIZE 32

namespace magmadnn {
namespace internal {

template <typename T>
__global__ void kernel_reduce_sum_grad_scalar_to_vector_device(T *grad, T *out, unsigned int out_size) {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < out_size; i += stride) {
        out[i] = grad[0];
    }
}

template <typename T>
__global__ void kernel_reduce_sum_grad_vector_to_matrix_device(T *grad, T *out, unsigned int n_rows, unsigned int n_cols, int axis) {
    unsigned int x_idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int x_stride = blockDim.x * gridDim.x;
    unsigned int y_idx = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int y_stride = blockDim.y * gridDim.y;

    for (unsigned int r = x_idx; r < n_rows; r += x_stride) {
        for (unsigned int c = y_idx; c < n_cols; c += y_stride) {
            out[r*n_cols + c] = (axis == 1) ? grad[c] : grad[r];
        }
    }
}

template <typename T>
void reduce_sum_grad_device(Tensor<T> *grad, int axis, Tensor<T> *out) {

    /* the gradient of reduce sum, should repeat grad along the axis that was reduced.
        Let a0, ... , an  be the shape of reduce_sum's input (also 'out'). Then
        grad should be of shape a0, ... , 1, ... an, where 1 is at index='axis'. */
    const std::vector<unsigned int>& grad_shape = grad->get_shape();
    const std::vector<unsigned int>& out_shape = out->get_shape();
    unsigned int n_grad_axes = grad_shape.size();
    unsigned int n_out_axes = out_shape.size();
    unsigned int out_size = out->get_size();

    /* TODO -- modularize this into one single algorithm */
    if (n_grad_axes == 1 && grad->get_size() == 1) {
        /* grad is a scalar -- fill out with the value */

        kernel_reduce_sum_grad_scalar_to_vector_device<<<(out_size+BLK_SIZE-1)/BLK_SIZE,BLK_SIZE>>> (grad->get_ptr(), out->get_ptr(), out_size);

    } else if (n_grad_axes == 2 && (grad_shape[0] == 1 || grad_shape[1] == 1)) {
        /* grad gets repeated along each row of out */

        unsigned int n_rows = out_shape[0];
        unsigned int n_cols = out_shape[1];

        dim3 block (BLK_2D_SIZE, BLK_2D_SIZE);
        dim3 grid;
        grid.x = (n_rows + block.x - 1) / block.x;
        grid.y = (n_cols + block.y - 1) / block.y;
        kernel_reduce_sum_grad_vector_to_matrix_device<<<grid,block>>> (grad->get_ptr(), out->get_ptr(), n_rows, n_cols, axis);

    } else {
        /* use math::tile for the more general case */
        math::tile(grad, out, out->get_shape(axis), axis);
    }
}
template void reduce_sum_grad_device(Tensor<int> *grad, int axis, Tensor<int> *out);
template void reduce_sum_grad_device(Tensor<float> *grad, int axis, Tensor<float> *out);
template void reduce_sum_grad_device(Tensor<double> *grad, int axis, Tensor<double> *out);
    
}   // namespace internal
}   // namespace magmadnn

#undef BLK_SIZE
#undef BLK_2D_SIZE
