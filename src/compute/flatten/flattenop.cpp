
#include "compute/flatten/flattenop.h"

namespace magmadnn {
namespace op {

template <typename T>
FlattenOp<T>::FlattenOp(Operation<T> *input, bool copy, bool needs_grad)
: Operation<T>::Operation({input}, needs_grad), input(input), copy(copy) {
    /* setup code in here */

    unsigned int batch_size = input->get_output_shape(0);
    unsigned int flattened_size = input->get_output_size() / batch_size;
    this->output_shape = {batch_size, flattened_size};

    this->mem_type = input->get_memory_type();
    this->name = "Flatten";

    this->input_tensor = input->get_output_tensor();
    this->output_tensor = new Tensor<T> (this->output_shape, {NONE, {}}, this->mem_type);
}

template <typename T>
Tensor<T> *FlattenOp<T>::_eval(bool recompute) {
    /* eval code in here ... */

    input_tensor = input->eval(recompute);
    this->output_tensor->copy_from(*input_tensor);
    this->output_tensor->reshape(this->output_shape);

    return this->output_tensor;
}

template <typename T>
Tensor<T> *FlattenOp<T>::_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad) {
    /* return gradient in here ... */
    Tensor<T> *out = this->_grad_cache[(uintptr_t)var];

    if (out == NULL) {
        out = new Tensor<T> (this->output_shape, {NONE, {}}, this->mem_type);
        this->_grad_cache[(uintptr_t)var] = out;
    }

    out->copy_from(*grad);
    out->reshape(input_tensor->get_shape());

    return out;
}

template class FlattenOp<int>;
template class FlattenOp<float>;
template class FlattenOp<double>;


template <typename T>
FlattenOp<T> *flatten(Operation<T> *input, bool copy, bool needs_grad) {
    return new FlattenOp<T>(input, copy, needs_grad);
}
template FlattenOp<int> *flatten(Operation<int> *input, bool copy, bool needs_grad);
template FlattenOp<float> *flatten(Operation<float> *input, bool copy, bool needs_grad);
template FlattenOp<double> *flatten(Operation<double> *input, bool copy, bool needs_grad);


}   // namespace op
}   // namespace magmadnn