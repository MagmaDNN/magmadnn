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
#include "optimizer/gradientdescent/gradientdescent_internal.h"

namespace magmadnn {
namespace optimizer {

template <typename T>
class GradientDescent : public Optimizer<T> {
public:
    GradientDescent(op::Operation<T> *_obj_func, T learning_rate);

    virtual void minimize(const std::vector<op::Operation<T> *>& wrt);

protected:
    virtual void update(op::Operation<T> *var, op::Operation<T> *grad);

    T learning_rate;
    op::GradTable<T> table;
};

}   // namespace optimizer
}   // namespace magmadnn