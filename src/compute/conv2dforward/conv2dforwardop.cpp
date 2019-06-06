
#include "compute/conv2dforward/conv2dforwardop.h"

namespace magmadnn {
namespace op {

template <typename T>
Conv2DForwardOp<T>::Conv2DForwardOp(Operation<T> *input, bool copy, bool needs_grad)
: Operation<T>::Operation({input}, needs_grad), input(input), copy(copy) {
    /* setup code in here */
    this->output_shape = input->get_output_shape();
    this->mem_type = input->get_memory_type();

    if (copy) {
        this->output_tensor = new Tensor<T> (this->output_shape, {NONE, {}}, this->mem_type);
    }
}

template <typename T>
Tensor<T> *Conv2DForwardOp<T>::_eval(bool recompute) {
    
    input_tensor = input->eval(recompute);

    

    return this->output_tensor;
}

template <typename T>
Tensor<T> *Conv2DForwardOp<T>::_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad) {
    /* return gradient in here ... */
    return grad;
}

template class Conv2DForwardOp<int>;
template class Conv2DForwardOp<float>;
template class Conv2DForwardOp<double>;


template <typename T>
Conv2DForwardOp<T> *conv2dforward(Operation<T> *input, bool copy, bool needs_grad) {
    return new Conv2DForwardOp<T>(input, copy, needs_grad);
}
template Conv2DForwardOp<int> *conv2dforward(Operation<int> *input, bool copy, bool needs_grad);
template Conv2DForwardOp<float> *conv2dforward(Operation<float> *input, bool copy, bool needs_grad);
template Conv2DForwardOp<double> *conv2dforward(Operation<double> *input, bool copy, bool needs_grad);


}   // namespace op
}   // namespace magmadnn