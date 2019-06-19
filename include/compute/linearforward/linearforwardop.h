
#pragma once

#include "compute/operation.h"
#include "tensor/tensor.h"
#include "math/matmul.h"

namespace magmadnn {
namespace op {

template <typename T>
class LinearForwardOp : public Operation<T> {
public:
	LinearForwardOp(Operation<T> *input, Operation<T> *weights, bool copy=true, bool needs_grad=true);
	
	std::string to_string() { return "LinearForward(" + input->to_string() + ", " + weights->to_string() + ")"; }
protected:
	Tensor<T> *_eval(bool recompute);
	Tensor<T> *_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad);

	Operation<T> *input, *weights;
	Tensor<T> *input_tensor, *weights_tensor;

	bool copy;

};

/** Defines the forward and backward pass of a fully connected layer.
 * @tparam T int float double
 * @param input tensor
 * @param weights tensor
 * @param copy 
 * @param needs_grad 
 * @return LinearForwardOp<T>* An operation that calculates the LinearForward operation.
 */
template <typename T>
LinearForwardOp<T>* linearforward(Operation<T> *input, Operation<T> *weights, bool copy=true, bool needs_grad=true);

} // namespace op
} // namespace magmadnn