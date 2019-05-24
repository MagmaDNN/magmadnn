
#pragma once

#include "compute/operation.h"
#include "tensor/tensor.h"
#include "compute/reducesum/reducesum_internal.h"

namespace magmadnn {
namespace op {

template <typename T>
class ReduceSumOp : public Operation<T> {
public:
	ReduceSumOp();

	Tensor<T> *eval();
	Operation<T> *grad(Operation<T> *consumer, Operation<T> *var, Operation<T> *grad);
	
	std::string to_string() { return ""; }
protected:

};

template <typename T>
ReduceSumOp<T>* reducesum();

} // namespace op
} // namespace magmadnn