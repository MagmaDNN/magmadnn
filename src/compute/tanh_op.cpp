/**
 * @file tanh_op.cpp
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-23
 * 
 * @copyright Copyright (c) 2019
 */
#include "compute/tanh_op.h"

namespace skepsi {
namespace op {

template <typename T>
tensor<T>* tanh_op<T>::eval() {
    tensor<T> *x_tensor = x->eval();

    tensor<T> *ret;
    if (copy) {
        ret = new tensor<T> (x_tensor->get_shape(), x_tensor->get_memory_type());
        ret->get_memory_manager()->copy_from(*(x_tensor->get_memory_manager()));
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