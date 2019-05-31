
#include "compute/transpose/transposeop.h"

namespace magmadnn {
namespace op {

template <typename T>
TransposeOp<T>::TransposeOp(Operation<T> *x, bool copy, bool needs_grad)
: Operation<T>::Operation({x}, needs_grad), x(x), copy(copy) {

    assert( OP_IS_MATRIX(x) );

    this->output_shape = {x->get_output_shape(1), x->get_output_shape(0)};
    this->mem_type = x->get_memory_type();

    if (copy) {
        this->ret = new Tensor<T> (this->output_shape, {NONE, {}}, this->mem_type);
    } else {
        std::fprintf(stderr, "Cannot transpose into same tensor.\n");
    }
}

template <typename T>
Tensor<T> *TransposeOp<T>::eval(bool recompute) {
    if (!recompute && this->ret != NULL) {
        return this->ret;
    }

    x_tensor = x->eval(recompute);

    internal::transpose_full(x_tensor, this->ret);

    return this->ret;
}

template <typename T>
Operation<T> *TransposeOp<T>::grad(Operation<T> *consumer, Operation<T> *var, Operation<T> *grad) {
    return transpose(grad, true, false);
}

template class TransposeOp<int>;
template class TransposeOp<float>;
template class TransposeOp<double>;


template <typename T>
TransposeOp<T> *transpose(Operation<T> *x, bool copy, bool needs_grad) {
    return new TransposeOp<T>(x, copy, needs_grad);
}
template TransposeOp<int> *transpose(Operation<int> *x, bool copy, bool needs_grad);
template TransposeOp<float> *transpose(Operation<float> *x, bool copy, bool needs_grad);
template TransposeOp<double> *transpose(Operation<double> *x, bool copy, bool needs_grad);


}   // namespace op
}   // namespace magmadnn
