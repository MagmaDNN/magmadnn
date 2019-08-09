
#pragma once

#include <vector>
#include "compute/compute_graph.h"
#include "compute/operation.h"
#include "math/batchnorm.h"
#include "tensor/tensor.h"

#if defined(_HAS_CUDA_)
#include "cudnn.h"
#endif

namespace magmadnn {
namespace op {

class BatchNormOp : public Operation {
   public:
    BatchNormOp(Operation *input);

    std::string to_string() const override { return "BatchNorm(" + input->to_string() + ")"; }

   protected:
    Tensor &_eval(bool recompute) override;
    Tensor &_grad(Operation *consumer, Operation *var, const Tensor &grad) override;

    Operation *input;

    unsigned int num_calls;
    Tensor bn_scale;
    Tensor bn_scale_diff;
    Tensor bn_bias;
    Tensor bn_bias_diff;
    Tensor running_mean;
    Tensor running_variance;
    Tensor saved_mean;
    Tensor saved_variance;

#if defined(_HAS_CUDA_)
    void init_settings();

    math::cudnn_batchnorm_settings_t settings;
#endif

    bool copy;
};

inline Operation *batchnorm(Operation *input) { return default_graph.add_operation<BatchNormOp>(input); }

}  // namespace op
}  // namespace magmadnn