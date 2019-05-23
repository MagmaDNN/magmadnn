/**
 * @file sigmoidop.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-23
 * 
 * @copyright Copyright (c) 2019
 */
#include "compute/sigmoid/sigmoidop.h"

namespace magmadnn {
namespace op {

template <typename T>
SigmoidOp<T>::SigmoidOp(Operation<T> *x, bool copy, bool fast) : 
    Operation<T>::Operation({x}), x(x), copy(copy), fast(fast) {

    this->output_shape = x->get_output_shape();
    this->mem_type = x->get_memory_type();
    
    /* create copy when tree is created, not at evaluation time. This avoids allocating memory when
       evaluating a compute tree. */
    if (copy) {
        this->ret = new Tensor<T> (x->get_output_shape(), x->get_memory_type());
    }
}

template <typename T>
Tensor<T>* SigmoidOp<T>::eval() {
    x_tensor = x->eval();

    /* ret was created in constructor, now just copy evaluated x_tensor into it */
    if (copy) {
        this->ret->copy_from(*x_tensor);
    } else {
        this->ret = x_tensor;
    }

    internal::sigmoid_full(this->ret, fast);
    
    return this->ret;
}

template <typename T>
Operation<T> *SigmoidOp<T>::grad(Operation<T> *consumer, Operation<T> *var, Operation<T> *grad) {
    /* sigmoid grad is   grad * output * (1-output)  */
    Operation<T> *output = op::var<T>(x->to_string(), x_tensor);
    Operation<T> *c = add<T>(scalar<T>("1", 1.0f, this->mem_type), negative<T>(output, false, false));
    return product<T>(grad, product<T>(output, c, false), false);
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
}   // namespace magmadnn