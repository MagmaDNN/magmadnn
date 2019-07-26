/**
 * @file gradientdescent.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-29
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include <map>
#include "compute/gradients.h"
#include "compute/gradtable.h"
#include "compute/operation.h"
#include "math/optimizer_math/sgd_momentum.h"
#include "optimizer/gradientdescent/gradientdescent_internal.h"
#include "optimizer/optimizer.h"

namespace magmadnn {
namespace optimizer {

template <typename T>
class GradientDescent : public Optimizer<T> {
   public:
    GradientDescent(T learning_rate, T momentum);

    virtual void minimize(op::Operation<T> *obj_func, const std::vector<op::Operation<T> *> &wrt);

    void set_learning_rate(T learning_rate) { this->learning_rate = learning_rate; }
    T get_learning_rate() { return this->learning_rate; }

    void set_momentum(T momentum) { this->momentum = momentum; }
    T get_momentum() { return this->momentum; }

   protected:
    virtual void update(op::Operation<T> *var, Tensor<T> *grad);

    T learning_rate;
    T momentum;
    op::GradTable<T> table;
    std::map<op::Operation<T> *, Tensor<T> *> momentum_table;
};

}  // namespace optimizer
}  // namespace magmadnn