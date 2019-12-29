#pragma once

#include "compute/operation.h"
#include "compute/reducesum/reducesum_internal.h"
#include "magmadnn/utilities_internal.h"
#include "math/reduce_sum.h"
#include "tensor/tensor.h"

namespace magmadnn {
namespace op {

template <typename T>
class ReduceSumOp : public Operation<T> {
   public:
    ReduceSumOp(Operation<T> *x, int axis, bool copy = true, bool needs_grad = true);

    virtual ~ReduceSumOp();

    std::string to_string() { return "ReduceSum( " + x->to_string() + " )"; }

   protected:
    Tensor<T> *_eval(bool recompute = true);
    Tensor<T> *_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad);

    Operation<T> *x;
    Tensor<T> *x_tensor;

    Tensor<T> *ones;

#if defined(MAGMADNN_HAVE_CUDA)
    math::reduce_sum_cudnn_settings_t reduce_settings;
#endif

    int axis;
    bool copy;
};

template <typename T>
ReduceSumOp<T> *reducesum(Operation<T> *x, int axis, bool copy = true, bool needs_grad = true);

}  // namespace op
}  // namespace magmadnn
