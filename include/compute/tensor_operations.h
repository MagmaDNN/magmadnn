/**
 * @file tensor_operators.h
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-18
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once
#include <vector>
#include "operation.h"
#include "tensor/tensor.h"

namespace skepsi {


template <typename T>
class add_nocopy : public operation {
public:
	add_nocopy(operation<T>& a, operation<T>& b) : operation<T>::operation({a, b}) {
		assert( children.size() == 2);
	}

	tensor<T>* eval() {
		tensor<T>* a = a->eval(children[0]);
		tensor<T>* b = b->eval(children[1]);

		assert( a->get_size() == b->get_size() );
		
		printf("adding a and b\n");
	} 
}



} // namespace skepsi
