
#pragma once

#include "compute/operation.h"
#include "tensor/tensor.h"
#include "compute/<#OPERATION_NAME_LOWER#>/<#OPERATION_NAME_LOWER#>_internal.h"

namespace magmadnn {
namespace op {

template <typename T>
class <#OPERATION_NAME#>Op : public Operation<T> {
public:
	<#OPERATION_NAME#>Op(Operation<T> *input, bool copy=true, bool needs_grad=true);

	Operation<T> *grad(Operation<T> *consumer, Operation<T> *var, Operation<T> *grad);
	
	std::string to_string() { return ""; }
protected:
	Tensor<T> *_eval(bool recompute);

	Operation<T> *input;
	Tensor<T> *input_tensor;

	bool copy;

};

template <typename T>
<#OPERATION_NAME#>Op<T>* <#OPERATION_NAME_LOWER#>(Operation<T> *input, bool copy=true, bool needs_grad=true);

} // namespace op
} // namespace magmadnn