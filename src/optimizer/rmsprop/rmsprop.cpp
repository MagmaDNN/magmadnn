/**
 * @file rmsprop.cpp
 * @author Sedrick Keh
 * @version 1.0
 * @date 2019-07-25
 *
 * @copyright Copyright (c) 2019
 */
#include "optimizer/rmsprop/rmsprop.h"

namespace magmadnn {
namespace optimizer {

template <typename T>
RMSProp<T>::RMSProp(T learning_rate, T decaying_factor)
    : Optimizer<T>::Optimizer(), learning_rate(learning_rate), decaying_factor(decaying_factor) {
    /* set the name of this Optimizer */
    this->_name = "RMSPropOptimizer";
}

template <typename T>
void RMSProp<T>::minimize(op::Operation<T> *obj_func, const std::vector<op::Operation<T> *> &wrt) {
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
void RMSProp<T>::update(op::Operation<T> *var, Tensor<T> *grad) {
    /* Initialize decaying_squares_average to 0 if doesn't exist */
    if (!this->decaying_squares_average.count(var)) {
        this->decaying_squares_average[var] = new Tensor<T>(grad->get_shape(), {ZERO, {}}, grad->get_memory_type());
    }

    Tensor<T> *var_tensor;
    var_tensor = var->eval(false);

    math::rmsprop(this->learning_rate, this->decaying_factor, this->decaying_squares_average[var], grad, var_tensor);
}

template class RMSProp<int>;
template class RMSProp<float>;
template class RMSProp<double>;

}  // namespace optimizer
}  // namespace magmadnn