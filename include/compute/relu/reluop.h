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

namespace skepsi {
namespace op {

template <typename T>
class ReluOp : public Operation<T> {
public:
    ReluOp(Operation<T> *x, bool copy=true);

    Tensor<T>* eval();
    Tensor<T>* grad();

    std::string to_string() { return "RELU( " + x->to_string() + " )"; }

protected:
    Operation<T> *x;
    Tensor<T> *ret;
    bool copy;
};


}   // namespace op
}   // namespace skepsi
