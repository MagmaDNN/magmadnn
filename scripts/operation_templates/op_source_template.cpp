
#include "compute/<#OPERATION_NAME_LOWER#>/<#OPERATION_NAME_LOWER#>op.h"

namespace magmadnn {
namespace op {

template <typename T>
<#OPERATION_NAME#>Op<T>::<#OPERATION_NAME#>Op() : Operation<T>::Operation() {
    /* setup code in here */
    this->output_shape = /* ... */
    this->mem_type = /* ... */
}

template <typename T>
Tensor<T> *<#OPERATION_NAME#>Op<T>::_eval() {
    /* eval code in here ... */
    return ret;
}

template <typename T>
Operation<T> *<#OPERATION_NAME#>Op<T>::grad(Operation<T> *consumer, Operation<T> *var, Operation<T> *grad) {
    /* return gradient in here ... */
    return grad;
}

template class <#OPERATION_NAME#>Op<int>;
template class <#OPERATION_NAME#>Op<float>;
template class <#OPERATION_NAME#>Op<double>;


template <typename T>
<#OPERATION_NAME#>Op<T> *<#OPERATION_NAME_LOWER#>() {
    return new <#OPERATION_NAME#>Op<T>();
}
template <#OPERATION_NAME#>Op<int> *<#OPERATION_NAME_LOWER#>();
template <#OPERATION_NAME#>Op<float> *<#OPERATION_NAME_LOWER#>();
template <#OPERATION_NAME#>Op<double> *<#OPERATION_NAME_LOWER#>();


}   // namespace op
}   // namespace magmadnn