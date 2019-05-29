/**
 * @file gradientdescent.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-29
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "optimizer/optimizer.h"
#include "compute/gradtable.h"
#include "compute/gradients.h"

namespace magmadnn {
namespace optimizer {

template <typename T>
class GradientDescent : public Optimizer<T> {
public:
    virtual GradientDescent(op::Operation<T> *_obj_func, T learning_rate);

    virtual void minimize(const std::vector<op::Operation<T> *>& wrt);

protected:
    T learning_rate;
    GradTable<T> table;
};

}   // namespace optimizer
}   // namespace magmadnn