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
#include "math/add.h"
#include "optimizer/gradientdescent/gradientdescent_internal.h"

namespace magmadnn {
namespace optimizer {

template <typename T>
class GradientDescent : public Optimizer<T> {
public:
    GradientDescent(T learning_rate);

    virtual void minimize(op::Operation<T> *obj_func, const std::vector<op::Operation<T> *>& wrt);

    void set_learning_rate(T learning_rate) { this->learning_rate = learning_rate; }
    T get_learning_rate() { return this->learning_rate; }

protected:
    virtual void update(op::Operation<T> *var, Tensor<T> *grad);

    T learning_rate;
    op::GradTable<T> table;
};

}   // namespace optimizer
}   // namespace magmadnn