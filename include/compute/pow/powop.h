
#pragma once

#include "compute/operation.h"
#include "tensor/tensor.h"
#include "compute/pow/pow_internal.h"
#include "math/pow.h"

namespace magmadnn {
namespace op {

template <typename T>
class PowOp : public Operation<T> {
public:
	PowOp(Operation<T> *input, int power, bool copy=true, bool needs_grad=true);

	
	std::string to_string() { return "POW("+input->to_string()+",)"; }
protected:
	Tensor<T> *_eval(bool recompute);
	Tensor<T> *_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad);

	Operation<T> *input;
	Tensor<T> *input_tensor;
	int power;

	bool copy;

};

template <typename T>
PowOp<T>* pow(Operation<T> *input, int power, bool copy=true, bool needs_grad=true);

} // namespace op
} // namespace magmadnn