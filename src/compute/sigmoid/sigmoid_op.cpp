/**
 * @file sigmoid_op.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-23
 * 
 * @copyright Copyright (c) 2019
 */
#include "compute/sigmoid/sigmoid_op.h"

namespace skepsi {
namespace op {

template <typename T>
tensor<T>* sigmoid_op<T>::eval() {
    tensor<T> *x_tensor = x->eval();

    tensor<T> *ret;
    if (copy) {
        ret = new tensor<T> (x_tensor->get_shape(), x_tensor->get_memory_type());
        ret->copy_from(*x_tensor);
    } else {
        ret = x_tensor;
    }

    internal::sigmoid_full(ret, fast);
    
    return ret;
}
template class sigmoid_op<int>;
template class sigmoid_op<float>;
template class sigmoid_op<double>;

template <typename T>
sigmoid_op<T>* sigmoid(operation<T> *x, bool copy, bool fast) {
    return new sigmoid_op<T> (x, fast, copy);
}
template sigmoid_op<int>* sigmoid(operation<int> *x, bool copy, bool fast);
template sigmoid_op<float>* sigmoid(operation<float> *x, bool copy, bool fast);
template sigmoid_op<double>* sigmoid(operation<double> *x, bool copy, bool fast);

}   // namespace op
}   // namespace skepsi