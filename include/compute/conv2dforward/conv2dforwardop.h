
#pragma once

#include "magmadnn/config.h"

#include "compute/conv2dforward/conv2dforward_internal.h"
#include "compute/operation.h"
#include "math/conv2d.h"
#include "tensor/tensor.h"

namespace magmadnn {
namespace op {

template <typename T>
class Conv2DForwardOp : public Operation<T> {
   public:
    Conv2DForwardOp(Operation<T> *input, Operation<T> *filter, int pad_h = 0, int pad_w = 0, int vertical_stride = 1,
                    int horizontal_stride = 1, int dilation_h = 1, int dilation_w = 1,
                    bool use_cross_correlation = true, bool needs_grad = true);
    ~Conv2DForwardOp();

    std::string to_string() { return "Conv2DForward(" + input->to_string() + ")"; }

   protected:
    Tensor<T> *_eval(bool recompute);
    Tensor<T> *_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad);
   
    void calculate_and_set_output_shape();

#if defined(MAGMADNN_HAVE_CUDA)
    void cuda_forward();
#endif
   
    void init_settings();

#if defined(MAGMADNN_HAVE_MKLDNN)
    void onednn_backward_data(Tensor<T> *grad, Tensor<T> *out);

    void onednn_backward_weights(Tensor<T> *grad, Tensor<T> *out);
   
    void onednn_forward();

    void onednn_init_settings();
#endif

    Operation<T> *input, *filter;
    Tensor<T> *input_tensor, *filter_tensor;

    int pad_h, pad_w, vertical_stride, horizontal_stride, dilation_h, dilation_w;
    bool use_cross_correlation;

#if defined(MAGMADNN_HAVE_CUDA)
    ::magmadnn::math::conv2d_cudnn_settings cudnn_settings;
#endif

#if defined(MAGMADNN_HAVE_MKLDNN)
   dnnl::engine dnnl_cpu_engine_;

   // Pooling DNNL primitive descriptor
   std::unique_ptr<dnnl::convolution_forward::primitive_desc> dnnl_fwd_pdesc_;

   // Pooling DNNL primitive
   std::unique_ptr<dnnl::convolution_forward> dnnl_fwd_;
#endif

};

template <typename T>
Conv2DForwardOp<T> *conv2dforward(Operation<T> *input, Operation<T> *filter, int pad_h = 0, int pad_w = 0,
                                  int vertical_stride = 1, int horizontal_stride = 1, int dilation_h = 1,
                                  int dilation_w = 1, bool use_cross_correlation = true, bool needs_grad = true);

}  // namespace op
}  // namespace magmadnn
