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

		assert( a->get_shape().size() == 2 );
		assert( a->get_size() == b->get_size() );

		for (int i = 0; i < (unsigned) b->get_shape()[0]; i++) {
			for (int j = 0; j < (unsigned) b->get_shape()[1]; j++) {
				b->set( {i,j}, a->get({i,j}) + b->get({i,j}) );
			}
		}
		return b;
	} 
};



} // namespace skepsi
