
#pragma once

#include "compute/operation.h"
#include "tensor/tensor.h"
#include "compute/negative/negative_internal.h"

namespace magmadnn {
namespace op {

template <typename T>
class NegativeOp : public Operation<T> {
public:
	NegativeOp(Operation<T> *x, bool copy, bool needs_grad);

	Operation<T> *grad(Operation<T> *consumer, Operation<T> *var, Operation<T> *grad);
	
	std::string to_string() { return "-" + x->to_string() + ""; }
protected:
	Tensor<T> *_eval(bool recompute=true);

	Operation<T> *x;
	Tensor<T> *x_tensor;

	bool copy;
};

template <typename T>
NegativeOp<T>* negative(Operation<T> *x, bool copy=true, bool needs_grad=true);

} // namespace op
} // namespace magmadnn
