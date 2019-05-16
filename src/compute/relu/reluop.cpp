/**
 * @file reluop.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-01
 * 
 * @copyright Copyright (c) 2019
 */
#include "compute/relu/reluop.h"

namespace skepsi {
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
    return ret;
}

template <typename T>
Tensor<T>* ReluOp<T>::grad() {
    return ret;
}
template class ReluOp<int>;
template class ReluOp<float>;
template class ReluOp<double>;

}   // namespace op
}   // namespace skepsi