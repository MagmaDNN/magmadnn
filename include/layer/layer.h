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
#include "tensor/tensor.h"

namespace skepsi {
namespace layer {

template <typename T>
class layer {
public:
	
	virtual void forward() = 0;
	virtual void backward() = 0;

	/** Get the pointer to the input tensor for this layer
	 * @return tensor<T>* 
	 */
	tensor<T> *get_input_tensor() { return input_tensor; }
	/** Get the pointer to the output tensor for this layer
	 * @return tensor<T>* 
	 */
	tensor<T> *get_output_tensor() { return output_tensor; }

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
	layer(std::vector<unsigned int> input_shape, tensor<T> *input_tensor) : 
		input_shape(input_shape), input_tensor(input_tensor) {}

	std::vector<unsigned int> input_shape;
	std::vector<unsigned int> output_shape;

	tensor<T> *input_tensor;
	tensor<T> *output_tensor;

};

}	// namespace layer
}	// namespace skepsi