/**
 * @file addop.h
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-18
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once
#include <vector>
#include "compute/operation.h"
#include "tensor/tensor.h"
#include "geadd_internal.h"

namespace magmadnn {
namespace op {

/**	An addition operation on two tensors.
 * @tparam T 
 */
template <typename T>
class AddOp : public Operation<T> {
public:
	/** Creates an Add Operation, which adds two tensors together.
	 * @param a a tensor
	 * @param b a tensor
	 * @param copy copy into new tensor
	 * @param needs_grad if this needs a gradient
	 */
	AddOp(Operation<T>* a, Operation<T>* b, bool copy=true, bool needs_grad=true);

	
	std::string to_string() { return "(" + a->to_string() + " + " + b->to_string() + ")"; }
protected:
	Tensor<T> *_eval(bool recompute=true);
	Tensor<T> *_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad);

	Operation<T>* a;
	Operation<T>* b;

	Tensor<T> *a_tensor;
	Tensor<T> *b_tensor;

	bool copy;
};

/** Returns a new add operation (@see AddOp<T>).
 * @tparam T 
 * @param a 
 * @param b 
 * @param copy If copy is true then it returns a new tensor, if false then b=a+b.
 * @return AddOp<T>* 
 */
template <typename T>
AddOp<T>* add(Operation<T> *a, Operation<T> *b, bool copy=true, bool needs_grad=true);

} // namespace op
} // namespace magmadnn
