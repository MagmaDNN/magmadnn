
#pragma once

#include "compute/negative/negative_internal.h"
#include "compute/operation.h"
#include "tensor/tensor.h"

namespace magmadnn {
namespace op {

template <typename T>
class NegativeOp : public Operation<T> {
   public:
    NegativeOp(Operation<T> *x, bool copy, bool needs_grad);

    std::string to_string() { return "-" + x->to_string() + ""; }

   protected:
    Tensor<T> *_eval(bool recompute = true);
    Tensor<T> *_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad);

    Operation<T> *x;
    Tensor<T> *x_tensor;

    bool copy;
};

template <typename T>
NegativeOp<T> *negative(Operation<T> *x, bool copy = true, bool needs_grad = true);

}  // namespace op
}  // namespace magmadnn
