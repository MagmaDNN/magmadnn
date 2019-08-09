
#pragma once

#include "compute/operation.h"
#include "tensor/tensor.h"

namespace magmadnn {
namespace op {

template <typename T>
class FlattenOp : public Operation<T> {
   public:
    FlattenOp(Operation<T> *input, bool copy = true, bool needs_grad = true);

    std::string to_string() { return "Flatten(" + input->to_string() + ")"; }

   protected:
    Tensor &_eval(bool recompute);
    Tensor<T> &_grad(Operation<T> *consumer, Operation<T> *var, const Tensor<T> &grad);

    Operation<T> *input;
    Tensor<T> *input_tensor;

    bool copy;
};

template <typename T>
FlattenOp<T> *flatten(Operation<T> *input, bool copy = true, bool needs_grad = true);

}  // namespace op
}  // namespace magmadnn