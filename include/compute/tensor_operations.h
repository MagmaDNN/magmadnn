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
class add_nocopy : public operation<T> {
public:
	add_nocopy(operation<T> *a, operation<T> *b) : operation<T>::operation({a, b}) {
		assert( this->children.size() == 2);
	}

	tensor<T>* eval() {
		tensor<T>* a = this->children[0]->eval();
		tensor<T>* b = this->children[1]->eval();

		assert( a->get_size() == b->get_size() );
		
		printf("adding a and b\n");
		
		for (unsigned int i = 0; i < a.get_size(); i++) {
			b->set()
		}
	} 
};



} // namespace skepsi
