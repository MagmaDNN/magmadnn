
#pragma once

#include "compute/operation.h"
#include "tensor/tensor.h"
#include "compute/conv2dforward/conv2dforward_internal.h"
#include "math/conv2d.h"

namespace magmadnn {
namespace op {

template <typename T>
class Conv2DForwardOp : public Operation<T> {
public:
	Conv2DForwardOp(Operation<T> *input, Operation<T> *filter, int pad_h=0, int pad_w=0, int vertical_stride=1, int horizontal_stride=1, int dilation_h=1, int dilation_w=1, bool use_cross_correlation=true, bool needs_grad=true);
	~Conv2DForwardOp();


	std::string to_string() { return "Conv2DForward(" + input->to_string() + ")"; }
protected:
	Tensor<T> *_eval(bool recompute);
	Tensor<T> *_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad);
	

	void init_settings();
	void calculate_and_set_output_shape();

	Operation<T> *input, *filter;
	Tensor<T> *input_tensor, *filter_tensor;

	int pad_h, pad_w, vertical_stride, horizontal_stride, dilation_h, dilation_w;
	bool use_cross_correlation;

	#if defined(_HAS_CUDA_)
	::magmadnn::math::conv2d_cudnn_settings cudnn_settings;
	#endif

};

template <typename T>
Conv2DForwardOp<T>* conv2dforward(Operation<T> *input, Operation<T> *filter, int pad_h=0, int pad_w=0, int vertical_stride=1, int horizontal_stride=1, int dilation_h=1, int dilation_w=1, bool use_cross_correlation=true, bool needs_grad=true);

} // namespace op
} // namespace magmadnn