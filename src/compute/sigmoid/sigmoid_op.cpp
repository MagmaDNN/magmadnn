/**
 * @file sigmoidop.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-23
 * 
 * @copyright Copyright (c) 2019
 */
#include "compute/sigmoid/sigmoidop.h"

namespace skepsi {
namespace op {

template <typename T>
SigmoidOp<T>::SigmoidOp(Operation<T> *x, bool copy, bool fast) : 
    Operation<T>::Operation({x}), x(x), copy(copy), fast(fast) {

    this->output_shape = x->get_output_shape();
    this->mem_type = x->get_memory_type();
    
    /* create copy when tree is created, not at evaluation time. This avoids allocating memory when
       evaluating a compute tree. */
    if (copy) {
        ret = new Tensor<T> (x->get_output_shape(), x->get_memory_type());
    }
}

template <typename T>
Tensor<T>* SigmoidOp<T>::eval() {
    Tensor<T> *x_tensor = x->eval();

    /* ret was created in constructor, now just copy evaluated x_tensor into it */
    if (copy) {
        ret->copy_from(*x_tensor);
    } else {
        ret = x_tensor;
    }

    internal::sigmoid_full(ret, fast);
    
    return ret;
}

template <typename T>
Tensor<T>* SigmoidOp<T>::grad() {
    return NULL;
}
template class SigmoidOp<int>;
template class SigmoidOp<float>;
template class SigmoidOp<double>;

template <typename T>
SigmoidOp<T>* sigmoid(Operation<T> *x, bool copy, bool fast) {
    return new SigmoidOp<T> (x, fast, copy);
}
template SigmoidOp<int>* sigmoid(Operation<int> *x, bool copy, bool fast);
template SigmoidOp<float>* sigmoid(Operation<float> *x, bool copy, bool fast);
template SigmoidOp<double>* sigmoid(Operation<double> *x, bool copy, bool fast);

}   // namespace op
}   // namespace skepsi