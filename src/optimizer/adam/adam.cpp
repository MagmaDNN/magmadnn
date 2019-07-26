/**
 * @file adam.cpp
 * @author Sedrick Keh
 * @version 1.0
 * @date 2019-07-25
 *
 * @copyright Copyright (c) 2019
 */
#include "optimizer/adam/adam.h"

namespace magmadnn {
namespace optimizer {

template <typename T>
Adam<T>::Adam(T learning_rate, T beta1, T beta2)
    : Optimizer<T>::Optimizer(),
      learning_rate(learning_rate),
      beta1(beta1),
      beta2(beta2),
      running_beta1(beta1),
      running_beta2(beta2) {
    /* set the name of this Optimizer */
    this->_name = "AdamOptimizer";
}

template <typename T>
void Adam<T>::minimize(op::Operation<T> *obj_func, const std::vector<op::Operation<T> *> &wrt) {
    typename std::vector<op::Operation<T> *>::const_iterator vit;

    this->_obj_func = obj_func;

    /* evaluate if need be */
    this->_obj_func->eval(false);

    /* build the gradients */
    this->table.clear();
    op::get_grad_table(wrt, this->_obj_func, this->table);

    /* now update each one */
    for (vit = wrt.begin(); vit != wrt.end(); vit++) {
        this->update((*vit), table.get(*vit));
    }

    /* update beta1 and beta2 */
    this->running_beta1 *= this->beta1;
    this->running_beta2 *= this->beta2;
}

template <typename T>
void Adam<T>::update(op::Operation<T> *var, Tensor<T> *grad) {
    /* Initialize first moments to 0 if doesn't exist */
    if (!this->first_moment.count(var)) {
        this->first_moment[var] = new Tensor<T>(grad->get_shape(), {ZERO, {}}, grad->get_memory_type());
    }

    /* Initialize second moments to 0 if doesn't exist */
    if (!this->second_moment.count(var)) {
        this->second_moment[var] = new Tensor<T>(grad->get_shape(), {ZERO, {}}, grad->get_memory_type());
    }

    Tensor<T> *var_tensor;
    var_tensor = var->eval(false);

    math::adam(this->learning_rate, this->beta1, this->beta2, this->running_beta1, this->running_beta2,
               this->first_moment[var], this->second_moment[var], grad, var_tensor);
}

template class Adam<int>;
template class Adam<float>;
template class Adam<double>;

}  // namespace optimizer
}  // namespace magmadnn