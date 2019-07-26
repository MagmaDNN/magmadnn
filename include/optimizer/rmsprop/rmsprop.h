/**
 * @file rmsprop.h
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
#include "math/optimizer_math/rmsprop.h"
#include "optimizer/optimizer.h"

namespace magmadnn {
namespace optimizer {

template <typename T>
class RMSProp : public Optimizer<T> {
   public:
    RMSProp(T learning_rate, T decaying_factor);

    virtual void minimize(op::Operation<T> *obj_func, const std::vector<op::Operation<T> *> &wrt);

    void set_learning_rate(T learning_rate) { this->learning_rate = learning_rate; }
    T get_learning_rate() { return this->learning_rate; }

    void set_decaying_factor(T decaying_factor) { this->decaying_factor = decaying_factor; }
    T get_decaying_factor() { return this->decaying_factor; }

   protected:
    virtual void update(op::Operation<T> *var, Tensor<T> *grad);

    T learning_rate;
    T decaying_factor;

    op::GradTable<T> table;
    std::map<op::Operation<T> *, Tensor<T> *> decaying_squares_average;
};

}  // namespace optimizer
}  // namespace magmadnn