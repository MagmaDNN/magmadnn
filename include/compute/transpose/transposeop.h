
#pragma once

#include "compute/operation.h"
#include "compute/transpose/transpose_internal.h"
#include "tensor/tensor.h"
#include "utilities_internal.h"

namespace magmadnn {
namespace op {

template <typename T>
class TransposeOp : public Operation<T> {
   public:
    TransposeOp(Operation<T> *x, bool copy = true, bool needs_grad = true);

    std::string to_string() { return x->to_string() + ".T"; }

   protected:
    Tensor &_eval(bool recompute = true);
    Tensor<T> &_grad(Operation<T> *consumer, Operation<T> *var, const Tensor<T> &grad);

    Operation<T> *x;
    Tensor<T> *x_tensor;

    bool copy;
};

template <typename T>
TransposeOp<T> *transpose(Operation<T> *x, bool copy = true, bool needs_grad = true);

}  // namespace op
}  // namespace magmadnn
