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
sigmoid_op<T>::sigmoid_op(operation<T> *x, bool copy, bool fast) : 
    operation<T>::operation({x}), x(x), copy(copy), fast(fast) {

    this->output_shape = x->get_output_shape();
    this->mem_type = x->get_memory_type();
    
    /* create copy when tree is created, not at evaluation time. This avoids allocating memory when
       evaluating a compute tree. */
    if (copy) {
        ret = new tensor<T> (x->get_output_shape(), x->get_memory_type());
    }
}

template <typename T>
tensor<T>* sigmoid_op<T>::eval() {
    tensor<T> *x_tensor = x->eval();

    /* ret was created in constructor, now just copy evaluated x_tensor into it */
    if (copy) {
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