
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
        this->output_tensor = new Tensor<T> (this->output_shape, {NONE, {}}, this->mem_type);
    } else {
        std::fprintf(stderr, "Cannot transpose into same tensor.\n");
    }
}

template <typename T>
Tensor<T> *TransposeOp<T>::_eval(bool recompute) {

    x_tensor = x->eval(recompute);

    internal::transpose_full(x_tensor, this->output_tensor);

    return this->output_tensor;
}

template <typename T>
Tensor<T> *TransposeOp<T>::_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad) {
    Tensor<T> *out;

    out = this->_grad_cache[(uintptr_t)var];
    if (out == NULL) {
        out = new Tensor<T> ({grad->get_shape(1), grad->get_shape(0)}, {NONE,{}}, this->mem_type);
        this->_grad_cache[(uintptr_t)var] = out;
    }

    internal::transpose_full(grad, out);

    return out;
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
