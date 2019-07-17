
#pragma once

#include "compute/div/divop.h"
#include "compute/log/log_internal.h"
#include "compute/operation.h"
#include "compute/product/productop.h"
#include "compute/variable.h"
#include "tensor/tensor.h"

namespace magmadnn {
namespace op {

template <typename T>
class LogOp : public Operation<T> {
   public:
    LogOp(Operation<T> *x, bool stable = false, bool copy = true, bool needs_grad = true);

    std::string to_string() { return "log( " + x->to_string() + " )"; }

   protected:
    Tensor<T> *_eval(bool recompute = true);
    Tensor<T> *_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad);

    Operation<T> *x;
    Tensor<T> *x_tensor;

    bool stable;
    bool copy;
};

template <typename T>
LogOp<T> *log(Operation<T> *x, bool stable = false, bool copy = true, bool needs_grad = true);

}  // namespace op
}  // namespace magmadnn
