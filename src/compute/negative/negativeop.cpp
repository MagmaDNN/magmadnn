
#include "compute/negative/negativeop.h"

namespace magmadnn {
namespace op {

template <typename T>
NegativeOp<T>::NegativeOp(Operation<T> *x, bool copy, bool needs_grad) 
    : Operation<T>::Operation({x}, needs_grad), x(x), copy(copy) {
    
    this->output_shape = x->get_output_shape();
    this->mem_type = x->get_memory_type();

    if (copy) {
        this->ret = new Tensor<T> (this->output_shape, {NONE,{}}, this->mem_type);
    }
}

template <typename T>
Tensor<T> *NegativeOp<T>::eval(bool recompute) {

    if (!recompute && this->ret != NULL) {
        return this->ret;
    }

    x_tensor = x->eval(recompute);

    if (!copy) this->ret = x_tensor;

    internal::negative_full(x_tensor, this->ret);
    
    return this->ret;
}

template <typename T>
Operation<T> *NegativeOp<T>::grad(Operation<T> *consumer, Operation<T> *var, Operation<T> *grad) {
    /* grad : -grad */
    return negative(grad, true, false);
}

template class NegativeOp<int>;
template class NegativeOp<float>;
template class NegativeOp<double>;


template <typename T>
NegativeOp<T> *negative(Operation<T> *x, bool copy, bool needs_grad) {
    return new NegativeOp<T>(x, copy, needs_grad);
}
template NegativeOp<int> *negative(Operation<int> *x, bool copy, bool needs_grad);
template NegativeOp<float> *negative(Operation<float> *x, bool copy, bool needs_grad);
template NegativeOp<double> *negative(Operation<double> *x, bool copy, bool needs_grad);


}   // namespace op
}   // namespace magmadnn
