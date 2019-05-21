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
ReluOp<T>::ReluOp(Operation<T> *x, bool copy) : Operation<T>::Operation({x}), x(x), copy(copy) {
    this->output_shape = x->get_output_shape();
    this->mem_type = x->get_memory_type();

    if (copy) {
        ret = new Tensor<T> (this->output_shape, this->mem_type);
    }
}

template <typename T>
Tensor<T>* ReluOp<T>::eval() {
    x_tensor = x->eval();
    
    if (!copy) ret = x_tensor;

    internal::relu_full(x_tensor, ret);

    return ret;
}

template <typename T>
Operation<T> *ReluOp<T>::grad(Operation<T> *consumer, Operation<T> *var, Operation<T> *grad) {
    return NULL;
}
template class ReluOp<int>;
template class ReluOp<float>;
template class ReluOp<double>;

}   // namespace op
}   // namespace magmadnn