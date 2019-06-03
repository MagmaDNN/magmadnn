/**
 * @file divop.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-23
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "compute/operation.h"
#include "tensor/tensor.h"
#include "compute/div/div_internal.h"

namespace magmadnn {

namespace internal {
enum div_op_t {
	TENSOR_DIV_TENSOR,
	SCALAR_DIV_TENSOR,
	TENSOR_DIV_SCALAR,
	VEC_DIV_SCALAR,
	SCALAR_DIV_VEC
};
}	// namespace internal

namespace op {


template <typename T>
class DivOp : public Operation<T> {
public:
	DivOp(Operation<T> *a, Operation<T> *b, bool copy, bool needs_grad);

	Operation<T> *grad(Operation<T> *consumer, Operation<T> *var, Operation<T> *grad);
	
	std::string to_string() { return "( " + a->to_string() + " / " + b->to_string() + " )"; }
protected:
	Tensor<T> *_eval(bool recompute=true);
	Operation<T> *a, *b;
	Tensor<T> *a_tensor, *b_tensor;

	internal::div_op_t op_type;

	bool copy;
};

template <typename T>
DivOp<T>* div(Operation<T> *a, Operation<T> *b, bool copy, bool needs_grad);

} // namespace op
} // namespace magmadnn
