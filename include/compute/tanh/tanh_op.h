/**
 * @file tanh_op.h
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-23
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once
#include <string>
#include "compute/operation.h"
#include "tanh_internal.h"

namespace skepsi {
namespace op {

/** Tanh Operation. Computes the tanh function element-wise.
 * @tparam T 
 */
template <typename T>
class tanh_op : public operation<T> {
public:
    tanh_op(operation<T> *x, bool copy=true) : operation<T>::operation({x}), x(x), copy(copy) {};

    tensor<T>* eval();

    std::string to_string() { return "TANH( " + x->to_string() + " )"; }

protected:
    operation<T> *x;
    bool copy;
};

/** Returns a new tanh operation, which is an element-wise tanh execution over x.
 * @tparam T 
 * @param x 
 * @param copy if true, then a new tensor is returned else x is overwritten.
 * @return tanh_op<T>* 
 */
template <typename T>
tanh_op<T>* tanh(operation<T> *x, bool copy=true);

}   // namespace op
}   // namespace skepsi