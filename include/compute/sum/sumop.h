/**
 * @file sumop.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-20
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once

#include <vector>
#include <cstdio>
#include "compute/operation.h"
#include "tensor/tensor.h"
#include "compute/sum/sum_internal.h"

namespace magmadnn {
namespace op {

template <typename T>
class SumOp : public Operation<T> {
public:
    SumOp(std::vector<Operation<T> *> ops, bool copy=true);

    Tensor<T> *eval();
    Operation<T> *grad(Operation<T> *consumer, Operation<T> *var, Operation<T> *grad);

    std::string to_string();

protected:
    std::vector<Operation<T> *> ops;
    bool copy;
};

template <typename T>
Operation<T> *sum(std::vector<Operation<T> *> ops, bool copy=true);

}   // namespace op
}   // namespace magmadnn