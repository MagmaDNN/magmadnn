
#include "compute/reducesum/reducesumop.h"

namespace magmadnn {
namespace op {

template <typename T>
ReduceSumOp<T>::ReduceSumOp(Operation<T> *x, int axis, bool copy, bool needs_grad)
    : Operation<T>::Operation({x}, needs_grad), x(x), axis(axis), copy(copy) {

    std::vector<unsigned int> const& x_output_shape = x->get_output_shape();
    this->mem_type = x->get_memory_type();

    /* don't allow an axis greater than size of shape */
    assert( axis < (int)x_output_shape.size() );

    if (x_output_shape.size() == 1 || axis == -1) {
        /* x is a 1D vector. simply sum elements */
        op_type = internal::ELEM_REDUCE;
        this->output_shape = {1};
    } else if (x_output_shape.size() == 2) {
        /* matrix reduction */
        if (axis == 0)  {
            op_type = internal::COL_REDUCE;
            /* the number of cols */
            this->output_shape = {x_output_shape.at(1)};
            ones = new Tensor<T> ({x_output_shape.at(0)}, {ONE,{}}, this->mem_type);
        } else {
            op_type = internal::ROW_REDUCE;
            /* the number of rows */
            this->output_shape = {x_output_shape.at(0)};
            ones = new Tensor<T> ({x_output_shape.at(1)}, {ONE,{}}, this->mem_type);
        }
    } else {
        op_type = internal::TENSOR_REDUCE;
        this->output_shape = x_output_shape;
        std::fprintf(stderr, "ReduceSum not available for general tensors with more than 2 axes.\n");
    }

    if (copy) {
        /* init to ones */
        this->output_tensor = new Tensor<T> (this->get_output_shape(), {ONE, {}}, this->mem_type);
    } else {
        std::fprintf(stderr, "Non-Copy ReduceSum not supported.\n");
    }
}

template <typename T>
Tensor<T> *ReduceSumOp<T>::_eval(bool recompute) {

    x_tensor = x->eval(recompute);

    if (!copy) { std::fprintf(stderr, "Non-Copy ReduceSum not supported.\n"); return this->output_tensor; }

    switch (op_type) {
        case internal::TENSOR_REDUCE:
            internal::tensor_reducesum_full(x_tensor, axis, this->output_tensor); break;
        case internal::COL_REDUCE:
            internal::col_reducesum_full(x_tensor, ones, this->output_tensor); break;
        case internal::ROW_REDUCE:
            internal::row_reducesum_full(x_tensor, ones, this->output_tensor); break;
        case internal::ELEM_REDUCE:
            internal::reducesum_full(x_tensor, this->output_tensor); break;
    }

    return this->output_tensor;
}

template <typename T>
Operation<T> *ReduceSumOp<T>::grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad) {
    /* repeat grad along specified axis */

    /* output_shape = x.shape */
    /* output_shape[axis] = 1 */
    /* tile_scaling = x.shape // output_shape */
    /* reshape grad to output_shape */
    /* tile grad tile_scaling */

    return grad;
}

template class ReduceSumOp<int>;
template class ReduceSumOp<float>;
template class ReduceSumOp<double>;


template <typename T>
ReduceSumOp<T> *reducesum(Operation<T> *x, int axis, bool copy, bool needs_grad) {
    return new ReduceSumOp<T> (x, axis, copy, needs_grad);
}
template ReduceSumOp<int> *reducesum(Operation<int> *x, int axis, bool copy, bool needs_grad);
template ReduceSumOp<float> *reducesum(Operation<float> *x, int axis, bool copy, bool needs_grad);
template ReduceSumOp<double> *reducesum(Operation<double> *x, int axis, bool copy, bool needs_grad);


}   // namespace op
}   // namespace magmadnn