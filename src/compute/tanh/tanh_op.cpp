/**
 * @file tanh_op.cpp
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-23
 * 
 * @copyright Copyright (c) 2019
 */
#include "compute/tanh/tanh_op.h"

namespace skepsi {
namespace op {

template <typename T>
tanh_op<T>::tanh_op(operation<T> *x, bool copy) : operation<T>::operation({x}), x(x), copy(copy) {
    
    this->output_shape = x->get_output_shape();
    this->mem_type = x->get_memory_type();

    /* create tensor here to avoid memory allocation at tree execution */
    if (copy) {
        ret = new tensor<T> (this->output_shape, this->mem_type);
    }
}

template <typename T>
tensor<T>* tanh_op<T>::eval() {
    x_tensor = x->eval();

    if (copy) {
        ret->copy_from(*x_tensor);
    } else {
        ret = x_tensor;
    }

    internal::tanh_full(ret);
    
    return ret;
}
template class tanh_op<int>;
template class tanh_op<float>;
template class tanh_op<double>;

template <typename T>
tanh_op<T>* tanh(operation<T> *x, bool copy) {
    return new tanh_op<T> (x, copy);
}
template tanh_op<int>* tanh(operation<int> *x, bool copy);
template tanh_op<float>* tanh(operation<float> *x, bool copy);
template tanh_op<double>* tanh(operation<double> *x, bool copy);

}   // namespace op
}   // namespace skepsi