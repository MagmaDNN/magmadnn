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
	AddOp(Operation<T>* a, Operation<T>* b, bool copy=true, bool needs_grad=true);

	Operation<T> *grad(Operation<T> *consumer, Operation<T> *var, Operation<T> *grad);
	
	std::string to_string() { return "(" + a->to_string() + " + " + b->to_string() + ")"; }
protected:
	Tensor<T> *_eval(bool recompute=true);

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
