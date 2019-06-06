
#pragma once

#include "compute/operation.h"
#include "tensor/tensor.h"
#include "compute/conv2dforward/conv2dforward_internal.h"

namespace magmadnn {
namespace op {

template <typename T>
class Conv2DForwardOp : public Operation<T> {
public:
	Conv2DForwardOp(Operation<T> *input, bool copy=true, bool needs_grad=true);

	Operation<T> *grad(Operation<T> *consumer, Operation<T> *var, Operation<T> *grad);
	
	std::string to_string() { return "Conv2DForward(" + input->to_string() + ")"; }
protected:
	Tensor<T> *_eval(bool recompute);

	Operation<T> *input;
	Tensor<T> *input_tensor;

	bool copy;

};

template <typename T>
Conv2DForwardOp<T>* conv2dforward(Operation<T> *input, bool copy=true, bool needs_grad=true);

} // namespace op
} // namespace magmadnn