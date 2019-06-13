
#include "compute/<#OPERATION_NAME_LOWER#>/<#OPERATION_NAME_LOWER#>op.h"

namespace magmadnn {
namespace op {

template <typename T>
<#OPERATION_NAME#>Op<T>::<#OPERATION_NAME#>Op(Operation<T> *input, bool copy, bool needs_grad)
: Operation<T>::Operation({input}, needs_grad), input(input), copy(copy) {
    /* setup code in here */
    this->output_shape = input->get_output_shape();
    this->mem_type = input->get_memory_type();
    this->name = "<#OPERATION_NAME#>";

    if (copy) {
        this->output_tensor = new Tensor<T> (this->output_shape, {NONE, {}}, this->mem_type);
    }
}

template <typename T>
Tensor<T> *<#OPERATION_NAME#>Op<T>::_eval(bool recompute) {
    /* eval code in here ... */
    return this->output_tensor;
}

template <typename T>
Tensor<T> *<#OPERATION_NAME#>Op<T>::_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad) {
    /* return gradient in here ... */
    Tensor<T> *out = this->_grad_cache[(uintptr_t)var];

    return out;
}

template class <#OPERATION_NAME#>Op<int>;
template class <#OPERATION_NAME#>Op<float>;
template class <#OPERATION_NAME#>Op<double>;


template <typename T>
<#OPERATION_NAME#>Op<T> *<#OPERATION_NAME_LOWER#>(Operation<T> *input, bool copy, bool needs_grad) {
    return new <#OPERATION_NAME#>Op<T>(input, copy, needs_grad);
}
template <#OPERATION_NAME#>Op<int> *<#OPERATION_NAME_LOWER#>(Operation<int> *input, bool copy, bool needs_grad);
template <#OPERATION_NAME#>Op<float> *<#OPERATION_NAME_LOWER#>(Operation<float> *input, bool copy, bool needs_grad);
template <#OPERATION_NAME#>Op<double> *<#OPERATION_NAME_LOWER#>(Operation<double> *input, bool copy, bool needs_grad);


}   // namespace op
}   // namespace magmadnn