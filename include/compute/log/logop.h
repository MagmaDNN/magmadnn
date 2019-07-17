
#pragma once

#include "compute/operation.h"
#include "tensor/tensor.h"
#include "compute/variable.h"
#include "compute/div/divop.h"
#include "compute/product/productop.h"
#include "compute/log/log_internal.h"

namespace magmadnn {
namespace op {

template <typename T>
class LogOp : public Operation<T> {
public:
	LogOp(Operation<T> *x, bool copy=true, bool needs_grad=true);
	
	std::string to_string() { return "log( " + x->to_string() + " )"; }
protected:
	Tensor<T> *_eval(bool recompute=true);
	Tensor<T> &_grad(Operation<T> *consumer, Operation<T> *var, const Tensor<T> &grad);

	Operation<T> *x;
	Tensor<T> *x_tensor;

	bool copy;
};

template <typename T>
LogOp<T>* log(Operation<T> *x, bool copy=true, bool needs_grad=true);

} // namespace op
} // namespace magmadnn
