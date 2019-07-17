/**
 * @file sumop.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-20
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include <cstdio>
#include <vector>
#include "compute/operation.h"
#include "compute/sum/sum_internal.h"
#include "tensor/tensor.h"
#include "utilities_internal.h"

namespace magmadnn {
namespace op {

template <typename T>
class SumOp : public Operation<T> {
   public:
    SumOp(std::vector<Operation<T> *> ops, bool copy = true);

    std::string to_string();

   protected:
    Tensor<T> *_eval(bool recompute = true);
    Tensor<T> &_grad(Operation<T> *consumer, Operation<T> *var, const Tensor<T> &grad);

    std::vector<Operation<T> *> ops;
    bool copy;
};

template <typename T>
Operation<T> *sum(std::vector<Operation<T> *> ops, bool copy = true);

}  // namespace op
}  // namespace magmadnn