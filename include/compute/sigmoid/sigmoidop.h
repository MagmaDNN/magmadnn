/**
 * @file sigmoidop.h
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-23
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once
#include <string>
#include "compute/operation.h"
#include "sigmoid_internal.h"

namespace skepsi {
namespace op {

/** Sigmoid Operation. Computes the element-wise sigmoid operation on a Tensor.
 * @tparam T 
 */
template <typename T>
class SigmoidOp : public Operation<T> {
public:
    SigmoidOp(Operation<T> *x, bool copy=true, bool fast=true);

    Tensor<T>* eval();

    std::string to_string() { return "SIGMOID( " + x->to_string() + " )"; }

protected:
    Operation<T> *x;
    Tensor<T> *ret;
    
    bool copy;
    bool fast;
};

/** Compute element-wise sigmoid on tensor x.
 * @tparam T 
 * @param x tensor to be computed with.
 * @param copy if true, a new tensor is allocated and returned. If false, x is overwritten.
 * @param fast if true, the x=1/(1+|x|) is computed instead of the normal sigmoid function.
 * @return SigmoidOp<T>* 
 */
template <typename T>
SigmoidOp<T>* sigmoid(Operation<T> *x, bool copy=true, bool fast=true);

}   // namespace op
}   // namespace skepsi