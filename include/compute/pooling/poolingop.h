
#pragma once

#include "compute/operation.h"
#include "math/pooling.h"
#include "tensor/tensor.h"

namespace magmadnn {
namespace op {

template <typename T>
class PoolingOp : public Operation<T> {
   public:
    PoolingOp(Operation<T> *input, int filter_h = 0, int filter_w = 0, int pad_h = 0, int pad_w = 0,
              int vertical_stride = 1, int horizontal_stride = 1, pooling_mode mode = MAX_POOL,
              bool propagate_nan = false, bool needs_grad = true);
    ~PoolingOp();

    std::string to_string() { return "Pooling(" + input->to_string() + ")"; }

   protected:
    Tensor &_eval(bool recompute);
    Tensor<T> &_grad(Operation<T> *consumer, Operation<T> *var, const Tensor<T> &grad);

    void init_settings();
    void calculate_and_set_output_shape();

    Operation<T> *input;
    Tensor<T> *input_tensor;

    int filter_h, filter_w, pad_h, pad_w, vertical_stride, horizontal_stride;
    pooling_mode mode;
    bool propagate_nan;

#if defined(_HAS_CUDA_)
    math::cudnn_pooling_settings_t settings;
#endif
};

template <typename T>
PoolingOp<T> *pooling(Operation<T> *input, int filter_h = 0, int filter_w = 0, int pad_h = 0, int pad_w = 0,
                      int vertical_stride = 1, int horizontal_stride = 1, pooling_mode mode = MAX_POOL,
                      bool propagate_nan = false, bool needs_grad = true);

}  // namespace op
}  // namespace magmadnn