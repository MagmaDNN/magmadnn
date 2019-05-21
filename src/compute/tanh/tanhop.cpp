/**
 * @file tanhop.cpp
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-23
 * 
 * @copyright Copyright (c) 2019
 */
#include "compute/tanh/tanhop.h"

namespace magmadnn {
namespace op {

template <typename T>
TanhOp<T>::TanhOp(Operation<T> *x, bool copy) : Operation<T>::Operation({x}), x(x), copy(copy) {
    
    this->output_shape = x->get_output_shape();
    this->mem_type = x->get_memory_type();

    /* create tensor here to avoid memory allocation at tree execution */
    if (copy) {
        ret = new Tensor<T> (this->output_shape, this->mem_type);
    }
}

template <typename T>
Tensor<T>* TanhOp<T>::eval() {
    x_tensor = x->eval();

    if (copy) {
        ret->copy_from(*x_tensor);
    } else {
        ret = x_tensor;
    }

    internal::tanh_full(ret);
    
    return ret;
}

template <typename T>
Operation<T> *TanhOp<T>::grad(Operation<T> *consumer, Operation<T> *var, Operation<T> *grad) {
    return NULL;
}
template class TanhOp<int>;
template class TanhOp<float>;
template class TanhOp<double>;

template <typename T>
TanhOp<T>* tanh(Operation<T> *x, bool copy) {
    return new TanhOp<T> (x, copy);
}
template TanhOp<int>* tanh(Operation<int> *x, bool copy);
template TanhOp<float>* tanh(Operation<float> *x, bool copy);
template TanhOp<double>* tanh(Operation<double> *x, bool copy);

}   // namespace op
}   // namespace magmadnn