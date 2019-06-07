/**
 * @file reluop.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-01
 * 
 * @copyright Copyright (c) 2019
 */
#include "compute/relu/reluop.h"

namespace magmadnn {
namespace op {

template <typename T>
ReluOp<T>::ReluOp(Operation<T> *x, bool copy, bool needs_grad)
    : Operation<T>::Operation({x}, needs_grad), x(x), copy(copy) {
    this->output_shape = x->get_output_shape();
    this->mem_type = x->get_memory_type();

    if (copy) {
        this->output_tensor = new Tensor<T> (this->output_shape, this->mem_type);
    }
}

template <typename T>
Tensor<T>* ReluOp<T>::_eval(bool recompute) {

    x_tensor = x->eval();
    
    if (!copy) this->output_tensor = x_tensor;

    internal::relu_full(x_tensor, this->output_tensor);

    return this->output_tensor;
}

template <typename T>
Tensor<T> *ReluOp<T>::_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad) {
    return NULL;
}
 
template class ReluOp<int>;
template class ReluOp<float>;
template class ReluOp<double>;

template <typename T>
ReluOp<T> *relu(Operation<T> *x, bool copy, bool needs_grad) {
    return new ReluOp<T> (x, copy, needs_grad);
}
template ReluOp<int> *relu(Operation<int> *x, bool copy, bool needs_grad);
template ReluOp<float> *relu(Operation<float> *x, bool copy, bool needs_grad);
template ReluOp<double> *relu(Operation<double> *x, bool copy, bool needs_grad);

}   // namespace op
}   // namespace magmadnn