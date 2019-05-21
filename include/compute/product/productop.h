/**
 * @file productop.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-21
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once

#include <vector>
#include "compute/operation.h"
#include "compute/variable.h"
#include "tensor/tensor.h"
#include "compute/product/product_internal.h"

namespace magmadnn {
namespace op {

template <typename T>
class ProductOp : public Operation<T> {
public:
	ProductOp(T alpha, Operation<T>* a, Operation<T>* b, bool copy=true);

	Tensor<T> *eval();
	Operation<T> *grad(Operation<T> *consumer, Operation<T> *var, Operation<T> *grad);
	
	std::string to_string() { return "(" + a->to_string() + " * " + b->to_string() + ")"; }
protected:
	Operation<T> *a;
	Operation<T> *b;

	Tensor<T> *a_tensor;
	Tensor<T> *b_tensor;

	T alpha;
	bool copy;
};

template <typename T>
ProductOp<T>* product(Operation<T> *a, Operation<T> *b, bool copy=true);

template <typename T>
ProductOp<T>* product(T alpha, Operation<T> *a, Operation<T> *b, bool copy=true);

} // namespace op
} // namespace magmadnn