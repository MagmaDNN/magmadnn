/**
 * @file gradientdescent.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-29
 *
 * @copyright Copyright (c) 2019
 */
#include "optimizer/gradientdescent/gradientdescent.h"

namespace magmadnn {
namespace optimizer {

template <typename T>
GradientDescent<T>::GradientDescent(T learning_rate, T momentum)
    : Optimizer<T>::Optimizer(), learning_rate(learning_rate), momentum(momentum) {
    /* set the name of this Optimizer */
    this->_name = "GradientDescentOptimizer";
}

template <typename T>
void GradientDescent<T>::minimize(op::Operation<T> *obj_func, const std::vector<op::Operation<T> *> &wrt) {
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
void GradientDescent<T>::update(op::Operation<T> *var, Tensor<T> *grad) {
    /* Initialize momentum value to 0 if doesn't exist */
    if (!this->momentum_table.count(var)) {
        this->momentum_table[var] = new Tensor<T>(grad->get_shape(), {ZERO, {}}, grad->get_memory_type());
    }

    Tensor<T> *var_tensor;

    var_tensor = var->eval(false);

    math::sgd_momentum(this->learning_rate, this->momentum, momentum_table[var], grad, var_tensor);
}

template class GradientDescent<int>;
template class GradientDescent<float>;
template class GradientDescent<double>;

}  // namespace optimizer
}  // namespace magmadnn