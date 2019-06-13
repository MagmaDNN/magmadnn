
#pragma once

#include "compute/operation.h"
#include "tensor/tensor.h"
#include "compute/scalarproduct/scalarproduct_internal.h"

namespace magmadnn {
namespace op {

/** Multiplies a tensor by a scalar
 * @tparam T numeric
 */
template <typename T>
class ScalarProductOp : public Operation<T> {
public:
	ScalarProductOp(T alpha, Operation<T> *x, bool copy=true, bool needs_grad=true);
	ScalarProductOp(Operation<T> *scalar, Operation<T> *x, bool copy=true, bool needs_grad=true);

	
	std::string to_string();
protected:
	Tensor<T> *_eval(bool recompute=true);
	Tensor<T> *_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad);

	T alpha;
	Operation<T> *scalar;
	Operation<T> *x;

	Tensor<T> *x_tensor;
	Tensor<T> *scalar_tensor;
	
	bool copy;
};

template <typename T>
ScalarProductOp<T> *scalarproduct(T alpha, Operation<T> *x, bool copy=true, bool needs_grad=true);

template <typename T>
ScalarProductOp<T> *scalarproduct(Operation<T> *scalar, Operation<T> *x, bool copy=true, bool needs_grad=true);

} // namespace op
} // namespace magmadnn
