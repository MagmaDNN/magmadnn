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
#include "compute/operation.h"
#include "tensor/tensor.h"
#include "geadd_internal.h"

namespace skepsi {
namespace op {

/**	An addition operation on two tensors.
 * @tparam T 
 */
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

/** Returns a new add operation (@see add_op<T>).
 * @tparam T 
 * @param a 
 * @param b 
 * @param copy If copy is true then it returns a new tensor, if false then b=a+b.
 * @return add_op<T>* 
 */
template <typename T>
add_op<T>* add(operation<T> *a, operation<T> *b, bool copy=true);

} // namespace op
} // namespace skepsi
