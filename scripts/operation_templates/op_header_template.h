
#pragma once

#include "compute/<#OPERATION_NAME_LOWER#>/<#OPERATION_NAME_LOWER#>_internal.h"
#include "compute/operation.h"
#include "tensor/tensor.h"

namespace magmadnn {
namespace op {

template <typename T>
class<#OPERATION_NAME #> Op : public Operation<T> {
   public:
    <#OPERATION_NAME #> Op(Operation<T> *input, bool copy = true, bool needs_grad = true);

    std::string to_string() { return ""; }

   protected:
    Tensor &_eval(borecompute);
    Tensor<T> *_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad);

    Operation<T> *input;
    Tensor<T> *input_tensor;

    bool copy;
};

template <typename T>
<#OPERATION_NAME #> Op<T> *<#OPERATION_NAME_LOWER #>(Operation<T> *input, bool copy = true, bool needs_grad = true);

}  // namespace op
}  // namespace magmadnn