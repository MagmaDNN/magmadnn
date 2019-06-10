
#include "compute/linearforward/linearforwardop.h"

namespace magmadnn {
namespace op {

template <typename T>
LinearForwardOp<T>::LinearForwardOp(Operation<T> *input, Operation<T> *weights, bool copy, bool needs_grad)
: Operation<T>::Operation({input}, needs_grad), input(input), weights(weights), copy(copy) {
    /* setup code in here */
    this->output_shape = {input->get_output_shape(0), weights->get_output_shape(1)};
    this->mem_type = input->get_memory_type();
    this->name = "LinearForward";

    if (copy) {
        this->output_tensor = new Tensor<T> (this->output_shape, {NONE, {}}, this->mem_type);
    }
}

template <typename T>
Tensor<T> *LinearForwardOp<T>::_eval(bool recompute) {
    input_tensor = input->eval(recompute);
    weights_tensor = weights->eval(recompute);

    /* XW */
    math::matmul((T)1, false, input_tensor, false, weights_tensor, (T)0, this->output_tensor);

    return this->output_tensor;
}

template <typename T>
Tensor<T> *LinearForwardOp<T>::_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad) {
    /* grad : GW^T */
    Tensor<T> *out = this->_grad_cache[(uintptr_t)var];

    weights_tensor = weights->eval(false);

    if (out == NULL) {
        out = new Tensor<T> ({grad->get_shape(0), weights_tensor->get_shape(0)}, {NONE,{}}, this->mem_type);
        this->_grad_cache[(uintptr_t)var] = out;
    }

    math::matmul((T)1, false, grad, true, weights_tensor, (T)0, out);

    return out;
}

template class LinearForwardOp<int>;
template class LinearForwardOp<float>;
template class LinearForwardOp<double>;


template <typename T>
LinearForwardOp<T> *linearforward(Operation<T> *input, Operation<T> *weights, bool copy, bool needs_grad) {
    return new LinearForwardOp<T>(input, weights, copy, needs_grad);
}
template LinearForwardOp<int> *linearforward(Operation<int> *input, Operation<int> *weights, bool copy, bool needs_grad);
template LinearForwardOp<float> *linearforward(Operation<float> *input, Operation<float> *weights, bool copy, bool needs_grad);
template LinearForwardOp<double> *linearforward(Operation<double> *input, Operation<double> *weights, bool copy, bool needs_grad);


}   // namespace op
}   // namespace magmadnn