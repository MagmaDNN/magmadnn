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
GradientDescent<T>::GradientDescent(op::Operation<T> *_obj_func, T learning_rate) : Optimizer<T>::Optimizer(_obj_func), learning_rate(learning_rate) {
    /* set the name of this Optimizer */
    this->_name = "GradientDescentOptimizer";
}

template <typename T>
void GradientDescent<T>::minimize(const std::vector<op::Operation<T> *>& wrt) {
    typename std::vector<op::Operation<T> *>::const_iterator vit;

    op::get_grad_table(wrt, this->_obj_func, this->table);
    
    for (vit = wrt.begin(); vit != wrt.end(); vit++) {
        this->update((*vit), table.get(*vit));
    }
}

template <typename T>
void GradientDescent<T>::update(op::Operation<T> *var, op::Operation<T> *grad) {
    Tensor<T> *var_tensor, *grad_tensor;

    var_tensor = var->eval();
    grad_tensor = grad->eval();

    internal::gradientdescent_update_internal(var_tensor, grad_tensor, this->learning_rate);
}

}   // namespace optimizer
}   // namespace magmadnn