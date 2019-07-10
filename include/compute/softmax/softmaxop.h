
#pragma once

#include "compute/operation.h"
#include "math/softmax.h"
#include "tensor/tensor.h"
#include "utilities_internal.h"

#if defined(_HAS_CUDA_)
#include "cudnn.h"
#endif

namespace magmadnn {
namespace op {

template <typename T>
class SoftmaxOp : public Operation<T> {
   public:
    SoftmaxOp(Operation<T> *input, bool copy = true, bool needs_grad = true);

    std::string to_string() { return "Softmax(" + input->to_string() + ")"; }

   protected:
    Tensor<T> *_eval(bool recompute);
    Tensor<T> *_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad);

    Operation<T> *input;
    Tensor<T> *input_tensor;

#if defined(_HAS_CUDA_)
    void init_settings();

    math::cudnn_softmax_settings_t settings;
    math::cudnn_softmax_grad_settings_t grad_settings;
#endif

    bool copy;
};

template <typename T>
SoftmaxOp<T> *softmax(Operation<T> *input, bool copy = true, bool needs_grad = true);

}  // namespace op
}  // namespace magmadnn