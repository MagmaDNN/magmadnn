
#pragma once

#include "compute/operation.h"
#include "tensor/tensor.h"
#include "utilities_internal.h"
#include "compute/crossentropy/crossentropy_internal.h"

namespace magmadnn {
namespace op {

template <typename T>
class CrossEntropyOp : public Operation<T> {
public:
	CrossEntropyOp(Operation<T> *x, Operation<T> *y, bool copy=true, bool needs_grad=true);
	~CrossEntropyOp();

	Operation<T> *grad(Operation<T> *consumer, Operation<T> *var, Operation<T> *grad);
	
	std::string to_string() { return "CrossEntropy(Softmax(" + x->to_string() + "), " + y->to_string() + ")"; }
protected:
	Tensor<T> *_eval(bool recompute=true);
	Operation<T> *x, *y;
	Tensor<T> *x_tensor, *y_tensor, *softmax;	/* scratch is used in the interal calc */

	bool copy;
};

template <typename T>
CrossEntropyOp<T>* crossentropy(Operation<T> *x, Operation<T> *y, bool copy=true, bool needs_grad=true);

} // namespace op
} // namespace magmadnn