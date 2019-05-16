/**
 * @file tanhop.h
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

namespace magmadnn {
namespace op {

/** Tanh Operation. Computes the tanh function element-wise.
 * @tparam T 
 */
template <typename T>
class TanhOp : public Operation<T> {
public:
    TanhOp(Operation<T> *x, bool copy=true);

    Tensor<T>* eval();

    std::string to_string() { return "TANH( " + x->to_string() + " )"; }

protected:
    Operation<T> *x;
    Tensor<T> *x_tensor;
    Tensor<T> *ret;
    bool copy;
};

/** Returns a new tanh operation, which is an element-wise tanh execution over x.
 * @tparam T 
 * @param x 
 * @param copy if true, then a new tensor is returned else x is overwritten.
 * @return TanhOp<T>* 
 */
template <typename T>
TanhOp<T>* tanh(Operation<T> *x, bool copy=true);

}   // namespace op
}   // namespace magmadnn