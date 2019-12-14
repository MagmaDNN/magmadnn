/**
 * @file gradientdescent.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-29
 *
 * @copyright Copyright (c) 2019
 */

#include "compute/gradients.h"
#include "compute/gradtable.h"
#include "compute/operation.h"
#include "math/optimizer_math/sgd_momentum.h"
#include "optimizer/distributed_momentum_sgd.h"
#include "optimizer/optimizer.h"

namespace magmadnn {
namespace optimizer {

   template <typename T>
   DistMomentumSGD<T>::DistMomentumSGD(T learning_rate, T momentum)
      : Optimizer<T>::Optimizer(), learning_rate(learning_rate),
      momentum(momentum) {

      this->_name = "DistMomentumSGD";
   }

   template <typename T>
   void DistMomentumSGD<T>::minimize(
         op::Operation<T> *obj_func,
         const std::vector<op::Operation<T> *> &wrt
         ) {
      
      typename std::vector<op::Operation<T> *>::const_iterator vit;

      this->_obj_func = obj_func;

      /* evaluate if need be */
      this->_obj_func->eval(false);

      /* build the gradients */
      this->table.clear();
      op::get_grad_table(wrt, this->_obj_func, this->table);

      /* now update each one */
      for (vit = wrt.begin(); vit != wrt.end(); vit++) {

         Tensor<T> *grad = this->table.get(*vit);

         // MPI_Allreduce(MPI_IN_PLACE, grad->get_ptr(), grad->get_size(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

         this->update((*vit), grad);
      }
   }

   template <typename T>
   void DistMomentumSGD<T>::update(
         op::Operation<T> *var,
         Tensor<T> *grad) {

      /* Initialize momentum value to 0 if doesn't exist */
      if (!this->momentum_table.count(var)) {
         this->momentum_table[var] = new Tensor<T>(grad->get_shape(), {ZERO, {}}, grad->get_memory_type());
      }

      Tensor<T> *var_tensor;

      var_tensor = var->eval(false);

      math::sgd_momentum(this->learning_rate, this->momentum, momentum_table[var], grad, var_tensor);
   }

   template class DistMomentumSGD<int>;
   template class DistMomentumSGD<float>;
   template class DistMomentumSGD<double>;

}  // namespace optimizer
}  // namespace magmadnn
