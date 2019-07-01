
#pragma once

#include "compute/operation.h"
#include "tensor/tensor.h"
#include "math/dropout.h"

#if defined(_HAS_CUDA_)
#include "cudnn.h"
#endif

namespace magmadnn {
namespace op {

template <typename T>
class DropoutOp : public Operation<T> {
public:
	DropoutOp(Operation<T> *input, float dropout_rate, bool copy=true, bool needs_grad=true);

	std::string to_string() { return "Dropout(" + input->to_string() + ")"; }
	
protected:
	Tensor<T> *_eval(bool recompute);
	Tensor<T> *_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad);

	Operation<T> *input;
	Tensor<T> *input_tensor;

	float dropout_rate;

	#if defined(_HAS_CUDA_)
	void init_settings();

	math::cudnn_dropout_settings_t settings;
	math::cudnn_dropout_grad_settings_t grad_settings;
	math::cudnn_dropout_shared_settings_t shared_settings;
	#endif

	bool copy;

};

template <typename T>
DropoutOp<T>* dropout(Operation<T> *input, float dropout_rate, bool copy=true, bool needs_grad=true);

} // namespace op
} // namespace magmadnn