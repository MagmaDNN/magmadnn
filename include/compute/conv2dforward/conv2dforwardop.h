
#pragma once

#include "compute/compute_graph.h"
#include "compute/operation.h"
#include "math/conv2d.h"
#include "tensor/tensor.h"

namespace magmadnn {
namespace op {

class Conv2DForwardOp : public Operation {
   public:
    Conv2DForwardOp(Operation *input, Operation *filter, int pad_h = 0, int pad_w = 0, int vertical_stride = 1,
                    int horizontal_stride = 1, int dilation_h = 1, int dilation_w = 1,
                    bool use_cross_correlation = true, bool needs_grad = true);
    ~Conv2DForwardOp();

    std::string to_string() const override { return "Conv2DForward(" + input->to_string() + ")"; }

   protected:
    Tensor &_eval(bool recompute) override;
    Tensor &_grad(Operation *consumer, Operation *var, const Tensor &grad) override;

    void init_settings();
    void calculate_and_set_output_shape();

    Operation *input, *filter;
    // Tensor *input_tensor, *filter_tensor;

    int pad_h, pad_w, vertical_stride, horizontal_stride, dilation_h, dilation_w;
    bool use_cross_correlation;

#if defined(_HAS_CUDA_)
    ::magmadnn::math::conv2d_cudnn_settings cudnn_settings;
#endif
};

inline Operation *conv2dforward(Operation *input, Operation *filter, int pad_h = 0, int pad_w = 0,
                                int vertical_stride = 1, int horizontal_stride = 1, int dilation_h = 1,
                                int dilation_w = 1, bool use_cross_correlation = true, bool needs_grad = true) {
    return default_graph.add_operation<Conv2DForwardOp>(input, filter, pad_h, pad_w, vertical_stride, horizontal_stride,
                                                        dilation_h, dilation_w, use_cross_correlation, needs_grad);
}

}  // namespace op
}  // namespace magmadnn