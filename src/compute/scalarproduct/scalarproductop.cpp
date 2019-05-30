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
    : Operation<T>::Operation({x}, needs_grad), alpha(alpha), scalar(NULL), x(x), copy(copy) {
    
    this->mem_type = x->get_memory_type();
    this->output_shape = x->get_output_shape();

    if (copy) {
        this->ret = new Tensor<T> (x->get_output_shape(), {NONE, {}}, x->get_memory_type());
    }
}

template <typename T>
ScalarProductOp<T>::ScalarProductOp(Operation<T> *scalar, Operation<T> *x, bool copy, bool needs_grad)
    : Operation<T>::Operation({scalar, x}, needs_grad), alpha((T)1), scalar(scalar), x(x), copy(copy) {

    assert( scalar->get_output_shape().size() == 1 && scalar->get_output_shape()[0] == 1 );
    this->mem_type = x->get_memory_type();
    this->output_shape = x->get_output_shape();

    if (copy) {
        this->ret = new Tensor<T> (x->get_output_shape(), {NONE, {}}, x->get_memory_type());
    }
}

template <typename T>
Tensor<T> *ScalarProductOp<T>::eval(bool recompute) {

    if (!recompute && this->ret != NULL) {
        return this->ret;
    }

    x_tensor = x->eval(recompute);
    
    if (scalar != NULL) {
        scalar_tensor = scalar->eval(recompute);
        scalar_tensor->get_memory_manager()->sync(true);
        alpha = scalar_tensor->get(0);
    }

    if (!copy) this->ret = x_tensor;

    internal::scalarproduct_full(alpha, x_tensor, this->ret);

    return this->ret;
}

template <typename T>
Operation<T> *ScalarProductOp<T>::grad(Operation<T> *consumer, Operation<T> *var, Operation<T> *grad) {
    if (scalar != NULL) {
        return scalarproduct(scalar, grad, false, false);
    } else {
        return scalarproduct(alpha, grad, false, false);
    }
}

template <typename T>
std::string ScalarProductOp<T>::to_string() {
    if (scalar != NULL) {
        return "( " + scalar->to_string() + " * " + x->to_string() + " )";
    } else {
        return "( __scalar * " + x->to_string() + " )";
    }
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


template <typename T>
ScalarProductOp<T> *scalarproduct(Operation<T> *scalar, Operation<T> *x, bool copy, bool needs_grad) {
    return new ScalarProductOp<T>(scalar, x, copy, needs_grad);
}
template ScalarProductOp<int> *scalarproduct(Operation<int> *scalar, Operation<int> *x, bool copy, bool needs_grad);
template ScalarProductOp<float> *scalarproduct(Operation<float> *scalar, Operation<float> *x, bool copy, bool needs_grad);
template ScalarProductOp<double> *scalarproduct(Operation<double> *scalar, Operation<double> *x, bool copy, bool needs_grad);


}   // namespace op
}   // namespace magmadnn
