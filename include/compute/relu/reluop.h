/**
 * @file reluop.h
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-05-01
 *
 * @copyright Copyright (c) 2019
 */
#pragma once
#include "compute/operation.h"
#include "compute/relu/relu_internal.h"
#include "math/relu.h"
#include "tensor/tensor.h"

namespace magmadnn {
namespace op {

template <typename T>
class ReluOp : public Operation<T> {
   public:
    ReluOp(Operation<T> *x, bool copy = true, bool needs_grad = true);
    virtual ~ReluOp();

    std::string to_string() { return "RELU( " + x->to_string() + " )"; }

   protected:
    Tensor<T> *_eval(bool recompute = true);
    Tensor<T> *_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad);

    Operation<T> *x;
    Tensor<T> *x_tensor;

#if defined(MAGMADNN_HAVE_CUDA)
    ::magmadnn::math::relu_cudnn_settings_t cudnn_settings;
#endif

    bool copy;
};

template <typename T>
ReluOp<T> *relu(Operation<T> *x, bool copy = true, bool needs_grad = true);

}  // namespace op
}  // namespace magmadnn
