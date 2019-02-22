/**
 * @file add_op.h
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-18
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once
#include <vector>
#include "operation.h"
#include "geadd_internal.h"
#include "tensor/tensor.h"

namespace skepsi {
namespace op {

template <typename T>
class add_op : public operation<T> {
public:
	add_op(operation<T>* a, operation<T>* b, bool copy=true) : 
		operation<T>::operation({a,b}), a(a), b(b), copy(copy) {}

	tensor<T>* eval();
	
	std::string to_string() { return "(" + a->to_string() + " + " + b->to_string() + ")"; }
protected:
	operation<T>* a;
	operation<T>* b;
	bool copy;
};

template <typename T>
add_op<T>* add(operation<T> *a, operation<T> *b, bool copy=true);

} // namespace op
} // namespace skepsi
