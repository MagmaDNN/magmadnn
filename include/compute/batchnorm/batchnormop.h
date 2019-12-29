
#pragma once

#include <vector>
#include "compute/operation.h"
#include "math/batchnorm.h"
#include "tensor/tensor.h"

#if defined(MAGMADNN_HAVE_CUDA)
#include "cudnn.h"
#endif

namespace magmadnn {
namespace op {

template <typename T>
class BatchNormOp : public Operation<T> {
   public:
    BatchNormOp(Operation<T> *input, bool needs_grad = true);

    virtual ~BatchNormOp();

    std::string to_string() { return "BatchNorm(" + input->to_string() + ")"; }

   protected:
    Tensor<T> *_eval(bool recompute);
    Tensor<T> *_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad);

    Operation<T> *input;
    Tensor<T> *input_tensor;

    unsigned int num_calls;
    Tensor<T> *bn_scale;
    Tensor<T> *bn_scale_diff;
    Tensor<T> *bn_bias;
    Tensor<T> *bn_bias_diff;
    Tensor<T> *running_mean;
    Tensor<T> *running_variance;
    Tensor<T> *saved_mean;
    Tensor<T> *saved_variance;

#if defined(MAGMADNN_HAVE_CUDA)
    void init_settings();

    math::cudnn_batchnorm_settings_t settings;
#endif

    bool copy;
};

template <typename T>
BatchNormOp<T> *batchnorm(Operation<T> *input, bool needs_grad = true);

}  // namespace op
}  // namespace magmadnn
