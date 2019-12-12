/**
 * @author Florent Lopez
 * @version 0.1
 * @date 2019-12-11
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "compute/gradtable.h"
#include "optimizer/optimizer.h"

namespace magmadnn {
namespace optimizer {

   template <typename T>
   class DistMomentumSGD : public Optimizer<T> {
   public:
      DistMomentumSGD(T learning_rate, T momentum);

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
