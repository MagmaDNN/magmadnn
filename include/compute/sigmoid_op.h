/**
 * @file sigmoid_op.h
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-23
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once
#include <string>
#include "operation.h"
#include "sigmoid_internal.h"

namespace skepsi {
namespace op {

template <typename T>
class sigmoid_op : public operation<T> {
public:
    sigmoid_op(operation<T> *x, bool copy=true, bool fast=true) : operation<T>::operation({x}), x(x), copy(copy), fast(fast) {};

    tensor<T>* eval();

    std::string to_string() { return "SIGMOID( " + x->to_string() + " )"; }

protected:
    operation<T> *x;
    bool copy;
    bool fast;
};

template <typename T>
sigmoid_op<T>* sigmoid(operation<T> *x, bool copy=true, bool fast=true);

}   // namespace op
}   // namespace skepsi