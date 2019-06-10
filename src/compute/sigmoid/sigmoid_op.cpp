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
        this->output_tensor = new Tensor<T> (this->output_shape, {NONE,{}}, this->mem_type);
    }

    this->_grad_cache[(uintptr_t)x] = NULL;
}

template <typename T>
Tensor<T>* SigmoidOp<T>::_eval(bool recompute) {

    x_tensor = x->eval(recompute);

    /* ret was created in constructor, now just copy evaluated x_tensor into it */
    if (copy) {
        this->output_tensor->copy_from(*x_tensor);
    } else {
        this->output_tensor = x_tensor;
    }

    internal::sigmoid_full(this->output_tensor, fast);
    
    return this->output_tensor;
}

template <typename T>
Tensor<T> *SigmoidOp<T>::_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad) {
    /* sigmoid grad is   grad * output * (1-output)  */

    Tensor<T> *out;
    Tensor<T> *output = this->eval(false);
    out = this->_grad_cache[(uintptr_t)var];

    if (out == NULL) {
        out = new Tensor<T> (this->output_shape, {NONE,{}}, this->mem_type);
        this->_grad_cache[(uintptr_t)var] = out;
    }

    internal::sigmoid_grad(output, grad, out);

    return out;
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