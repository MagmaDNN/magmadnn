/**
 * @file matmul_op.h
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-20
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once
#include <vector>
#include "operation.h"
#include "tensor/tensor.h"

namespace skepsi {
namespace op {

template <typename T>
class matmul_op : public operation<T> {
public:
	matmul_op(operation<T>* a, operation<T>* b, bool copy=false) : operation<T>::operation({a,b}), a(a), b(b) {}

	tensor<T>* eval();
	
	std::string to_string() { return "(" + a->to_string() + " * " + b->to_string() + ")"; }
protected:
	operation<T>* a;
	operation<T>* b;
};

template <typename T>
matmul_op<T>* matmul(operation<T> *a, operation<T> *b, bool copy=false);

} // namespace op
} // namespace skepsi
