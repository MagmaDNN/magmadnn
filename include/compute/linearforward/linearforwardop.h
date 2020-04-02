
#pragma once

#include "compute/operation.h"
#include "math/bias_add.h"
#include "math/matmul.h"
#include "math/reduce_sum.h"
#include "tensor/tensor.h"

namespace magmadnn {
namespace op {

template <typename T>
class LinearForwardOp : public Operation<T> {
   public:
    LinearForwardOp(Operation<T> *input, Operation<T> *weights, bool copy = true, bool needs_grad = true);
    LinearForwardOp(Operation<T> *input, Operation<T> *weights, Operation<T> *bias, bool copy = true,
                    bool needs_grad = true);
    virtual ~LinearForwardOp();

    std::string to_string() { return "LinearForward(" + input->to_string() + ", " + weights->to_string() + ")"; }

   protected:
    Tensor<T> *_eval(bool recompute);
    Tensor<T> *_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad);

    // Initialize linear forward operation and create output tensor
    void init_settings();

    void init_bias_settings(); /* init ones and bias_reduce_settings */

#if defined(MAGMADNN_HAVE_MKLDNN)
    void init_dnnl_settings();
#endif
   
    Operation<T> *input, *weights, *bias;
    Tensor<T> *input_tensor, *weights_tensor, *bias_tensor, *bias_ones;

    bool copy;
    bool use_bias;

#if defined(MAGMADNN_HAVE_CUDA)
    math::reduce_sum_cudnn_settings_t bias_reduce_settings;
#endif

#if defined(MAGMADNN_HAVE_MKLDNN)
   dnnl::engine dnnl_cpu_engine_;

   // Pooling DNNL primitive descriptor
   std::unique_ptr<dnnl::matmul::primitive_desc> dnnl_fwd_pdesc_;

   // Pooling DNNL primitive
   std::unique_ptr<dnnl::matmul> dnnl_fwd_;

#endif

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
LinearForwardOp<T> *linearforward(Operation<T> *input, Operation<T> *weights, bool copy = true, bool needs_grad = true);

/** Defines the forward and backward pass of a fully connected layer.
 * @tparam T int float double
 * @param input tensor
 * @param weights tensor
 * @param bias tensor
 * @param copy
 * @param needs_grad
 * @return LinearForwardOp<T>* An operation that calculates the LinearForward operation.
 */
template <typename T>
LinearForwardOp<T> *linearforward(Operation<T> *input, Operation<T> *weights, Operation<T> *bias, bool copy = true,
                                  bool needs_grad = true);

}  // namespace op
}  // namespace magmadnn
