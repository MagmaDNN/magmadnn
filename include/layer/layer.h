/**
 * @file layer.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-07
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once
#include <vector>
#include <string>
#include "compute/operation.h"

namespace skepsi {
namespace layer {

template <typename T>
class Layer {
public:
	
	virtual op::Operation<T>* out() {
		return output;
	}

	/** Get the pointer to the input tensor for this layer
	 * @return tensor<T>* 
	 */
	op::Operation<T> *get_input() { return input; }
	/** Get the pointer to the output tensor for this layer
	 * @return tensor<T>* 
	 */
	op::Operation<T> *get_output() { return output; }

	/** Returns a copy of the input shape as a vector
	 * @return std::vector<unsigned int> 
	 */
	std::vector<unsigned int> get_input_shape() const { return input_shape; }
	/** Returns a copy of the output shape as a vector
	 * @return std::vector<unsigned int> 
	 */
	std::vector<unsigned int> get_output_shape() const { return input_shape; }

	/** Gets the size of the i-th axis of the input tensor
	 * @param i axis
	 * @return std::vector<unsigned int> 
	 */
	std::vector<unsigned int> get_input_shape(unsigned int i) const {
		assert( i < input_shape.size() );
		return input_shape[i];
	}

	/** Gets the size of the i-th axis of the output tensor
	 * @param i axis
	 * @return std::vector<unsigned int> 
	 */
	std::vector<unsigned int> get_output_shape(unsigned int i) const {
		assert( i < output_shape.size() );
		return output_shape[i];
	}

protected:
	Layer(std::vector<unsigned int> input_shape, op::Operation<T> *input) : 
		input_shape(input_shape), input(input) {}

	std::vector<unsigned int> input_shape;
	std::vector<unsigned int> output_shape;

	op::Operation<T> *input;
	op::Operation<T> *output;

	std::string name;

};

}	// namespace layer
}	// namespace skepsi