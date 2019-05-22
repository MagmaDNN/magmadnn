/**
 * @file scalarproductop.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-22
 * 
 * @copyright Copyright (c) 2019
 */
#include "compute/scalarproduct/scalarproductop.h"

namespace magmadnn {
namespace op {

template <typename T>
ScalarProductOp<T>::ScalarProductOp(T alpha, Operation<T> *x, bool copy, bool needs_grad) 
    : Operation<T>::Operation({x}, needs_grad), alpha(alpha), x(x), copy(copy) {
    
    this->mem_type = x->get_memory_type();
    this->output_shape = x->get_output_shape();

    if (copy) {
        this->ret = new Tensor<T> (x->get_output_shape(), {NONE, {}}, x->get_memory_type());
    }
}

template <typename T>
Tensor<T> *ScalarProductOp<T>::eval() {
    x_tensor = x->eval();

    if (!copy) this->ret = x_tensor;

    internal::scalarproduct_full(alpha, x_tensor, this->ret);

    return this->ret;
}

template <typename T>
Operation<T> *ScalarProductOp<T>::grad(Operation<T> *consumer, Operation<T> *var, Operation<T> *grad) {
    return scalarproduct(alpha, grad, false, false);
}

template class ScalarProductOp<int>;
template class ScalarProductOp<float>;
template class ScalarProductOp<double>;


template <typename T>
ScalarProductOp<T> *scalarproduct(T alpha, Operation<T> *x, bool copy, bool needs_grad) {
    return new ScalarProductOp<T>(alpha, x, copy, needs_grad);
}
template ScalarProductOp<int> *scalarproduct(int alpha, Operation<int> *x, bool copy, bool needs_grad);
template ScalarProductOp<float> *scalarproduct(float alpha, Operation<float> *x, bool copy, bool needs_grad);
template ScalarProductOp<double> *scalarproduct(double alpha, Operation<double> *x, bool copy, bool needs_grad);


}   // namespace op
}   // namespace magmadnn
