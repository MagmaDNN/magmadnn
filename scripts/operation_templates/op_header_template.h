
#pragma once

#include "compute/operation.h"
#include "tensor/tensor.h"
#include "compute/<#OPERATION_NAME_LOWER#>/<#OPERATION_NAME_LOWER#>_internal.h"

namespace magmadnn {
namespace op {

template <typename T>
class <#OPERATION_NAME_FIRST_UPPER#>Op : public Operation<T> {
public:
	<#OPERATION_NAME_FIRST_UPPER#>Op();

	Tensor<T> *eval();
	Operation<T> *grad(Operation<T> *consumer, Operation<T> *var, Operation<T> *grad);
	
	std::string to_string() { return ""; }
protected:

};

template <typename T>
<#OPERATION_NAME_FIRST_UPPER#>Op<T>* <#OPERATION_NAME_LOWER#>();

template <typename T>
<#OPERATION_NAME_FIRST_UPPER#>Op<T>* <#OPERATION_NAME_LOWER#>();

} // namespace op
} // namespace magmadnn