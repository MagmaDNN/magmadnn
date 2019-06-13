
#include "compute/<#OPERATION_NAME_LOWER#>/<#OPERATION_NAME_LOWER#>_internal.h"

namespace magmadnn {
namespace internal {

template <typename T>
void <#OPERATION_NAME_LOWER#>_full(Tensor<T> *input, Tensor<T> *out) {}
template void <#OPERATION_NAME_LOWER#>_full(Tensor<int> *input, Tensor<int> *out);
template void <#OPERATION_NAME_LOWER#>_full(Tensor<float> *input, Tensor<float> *out);
template void <#OPERATION_NAME_LOWER#>_full(Tensor<double> *input, Tensor<double> *out);

template <typename T>
void <#OPERATION_NAME_LOWER#>_grad(Tensor<T> *grad, Tensor<T> *out) {}
template void <#OPERATION_NAME_LOWER#>_grad(Tensor<int> *input, Tensor<int> *out);
template void <#OPERATION_NAME_LOWER#>_grad(Tensor<float> *input, Tensor<float> *out);
template void <#OPERATION_NAME_LOWER#>_grad(Tensor<double> *input, Tensor<double> *out);

}   // namespace op
}   // namespace magmadnn