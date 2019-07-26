/**
 * @file adagrad.h
 * @author Sedrick Keh
 * @version 1.0
 * @date 2019-07-25
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include <map>
#include "compute/gradients.h"
#include "compute/gradtable.h"
#include "compute/operation.h"
#include "math/optimizer_math/adagrad.h"
#include "optimizer/optimizer.h"

namespace magmadnn {
namespace optimizer {

template <typename T>
class AdaGrad : public Optimizer<T> {
   public:
    AdaGrad(T learning_rate);

    virtual void minimize(op::Operation<T> *obj_func, const std::vector<op::Operation<T> *> &wrt);

    void set_learning_rate(T learning_rate) { this->learning_rate = learning_rate; }
    T get_learning_rate() { return this->learning_rate; }

   protected:
    virtual void update(op::Operation<T> *var, Tensor<T> *grad);

    T learning_rate;

    op::GradTable<T> table;
    std::map<op::Operation<T> *, Tensor<T> *> scaling_tensors;
};

}  // namespace optimizer
}  // namespace magmadnn