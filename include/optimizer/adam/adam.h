/**
 * @file adam.h
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
#include "math/optimizer_math/adam.h"
#include "optimizer/optimizer.h"

namespace magmadnn {
namespace optimizer {

template <typename T>
class Adam : public Optimizer<T> {
   public:
    Adam(T learning_rate, T beta1, T beta2);

    virtual void minimize(op::Operation<T> *obj_func, const std::vector<op::Operation<T> *> &wrt);

    void set_learning_rate(T learning_rate) { this->learning_rate = learning_rate; }
    T get_learning_rate() { return this->learning_rate; }

   protected:
    virtual void update(op::Operation<T> *var, Tensor<T> *grad);

    T learning_rate;
    T beta1;
    T beta2;
    T running_beta1;
    T running_beta2;

    op::GradTable<T> table;
    std::map<op::Operation<T> *, Tensor<T> *> first_moment;
    std::map<op::Operation<T> *, Tensor<T> *> second_moment;
};

}  // namespace optimizer
}  // namespace magmadnn