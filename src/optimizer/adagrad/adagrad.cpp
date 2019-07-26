/**
 * @file adagrad.cpp
 * @author Sedrick Keh
 * @version 1.0
 * @date 2019-07-25
 *
 * @copyright Copyright (c) 2019
 */
#include "optimizer/adagrad/adagrad.h"

namespace magmadnn {
namespace optimizer {

template <typename T>
AdaGrad<T>::AdaGrad(T learning_rate) : Optimizer<T>::Optimizer(), learning_rate(learning_rate) {
    /* set the name of this Optimizer */
    this->_name = "AdaGradOptimizer";
}

template <typename T>
void AdaGrad<T>::minimize(op::Operation<T> *obj_func, const std::vector<op::Operation<T> *> &wrt) {
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
}

template <typename T>
void AdaGrad<T>::update(op::Operation<T> *var, Tensor<T> *grad) {
    /* Initialize scaling_tensors to 0 if doesn't exist */
    if (!this->scaling_tensors.count(var)) {
        this->scaling_tensors[var] = new Tensor<T>(grad->get_shape(), {ZERO, {}}, grad->get_memory_type());
    }

    Tensor<T> *var_tensor;
    var_tensor = var->eval(false);

    math::adagrad(this->learning_rate, this->scaling_tensors[var], grad, var_tensor);
}

template class AdaGrad<int>;
template class AdaGrad<float>;
template class AdaGrad<double>;

}  // namespace optimizer
}  // namespace magmadnn