
#include "compute/<#OPERATION_NAME_LOWER#>/<#OPERATION_NAME_LOWER#>_internal.h"

namespace magmadnn {
namespace internal {
 
template <typename T>
__global__ void kernel_<#OPERATION_NAME_LOWER#>_full_device() {}
 
template <typename T>
void <#OPERATION_NAME_LOWER#>_full_device(Tensor<T> *input, Tensor<T> *out) {}
template void <#OPERATION_NAME_LOWER#>_full_device(Tensor<int> *input, Tensor<int> *out);
template void <#OPERATION_NAME_LOWER#>_full_device(Tensor<float> *input, Tensor<float> *out);
template void <#OPERATION_NAME_LOWER#>_full_device(Tensor<double> *input, Tensor<float> *out);

template <typename T>
__global__ void kernel_<#OPERATION_NAME_LOWER#>_grad_device() {}
 
template <typename T>
void <#OPERATION_NAME_LOWER#>_grad_device(Tensor<T> *grad, Tensor<T> *out) {}
template void <#OPERATION_NAME_LOWER#>_grad_device(Tensor<int> *input, Tensor<int> *out);
template void <#OPERATION_NAME_LOWER#>_grad_device(Tensor<float> *input, Tensor<float> *out);
template void <#OPERATION_NAME_LOWER#>_grad_device(Tensor<double> *input, Tensor<double> *out);
 
}   // namespace op
}   // namespace magmadnn