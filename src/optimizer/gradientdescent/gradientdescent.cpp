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
    _name = "GradientDescentOptimizer";
}

template <typename T>
void GradientDescent<T>::minimize(const std::vector<op::Operation<T> *>& wrt) {
    typename std::vector<op::Operation<T> *>::const_iterator vit;

    op::get_grad_table(wrt, _obj_func, table);
    
    for (vit = wrt.begin(); it != wrt.end(); it++) {

    }
}

}   // namespace optimizer
}   // namespace magmadnn