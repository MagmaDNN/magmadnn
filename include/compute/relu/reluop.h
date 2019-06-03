/**
 * @file reluop.h
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-05-01
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once
#include "compute/operation.h"
#include "compute/relu/relu_internal.h"
#include "tensor/tensor.h"

namespace magmadnn {
namespace op {

template <typename T>
class ReluOp : public Operation<T> {
public:
    ReluOp(Operation<T> *x, bool copy=true, bool needs_grad=true);

    Operation<T> *grad(Operation<T> *consumer, Operation<T> *var, Operation<T> *grad);

    std::string to_string() { return "RELU( " + x->to_string() + " )"; }

protected:
	Tensor<T> *_eval(bool recompute=true);

    Operation<T> *x;
    Tensor<T> *x_tensor;

    bool copy;
};

template <typename T>
ReluOp<T> *relu(Operation<T> *x, bool copy=true, bool needs_grad=true);


}   // namespace op
}   // namespace magmadnn
