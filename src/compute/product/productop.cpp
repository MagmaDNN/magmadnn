/**
 * @file productop.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-21
 * 
 * @copyright Copyright (c) 2019
 */
#include "compute/product/productop.h"

namespace magmadnn {
namespace op {

template <typename T>
ProductOp<T>::ProductOp(T alpha, Operation<T> *a, Operation<T> *b, bool copy, bool needs_grad)
    : Operation<T>::Operation({a,b}, needs_grad), alpha(alpha), a(a), b(b), copy(copy) {
    
    if (a->get_output_size() == 1) {
        op_type = internal::SCALAR_PROD_TENSOR;
        this->output_shape = b->get_output_shape();
    } else if (b->get_output_size() == 1) {
        op_type = internal::TENSOR_PROD_SCALAR;
        this->output_shape = a->get_output_shape();
    } else {
        assert( a->get_output_size() == b->get_output_size() );
        op_type = internal::TENSOR_PROD_TENSOR;
        this->output_shape = a->get_output_shape();
    }
    this->mem_type = a->get_memory_type();

    if (copy) {
        this->ret = new Tensor<T> (this->output_shape, {ONE, {}}, this->mem_type);
    }
}

template <typename T>
Tensor<T> *ProductOp<T>::eval(bool recompute) {

    if (!recompute && this->ret != NULL) {
        return this->ret;
    }

    a_tensor = a->eval(recompute);
    b_tensor = b->eval(recompute);
    
    if (!copy) {
        if (op_type == internal::TENSOR_PROD_SCALAR) {
            this->ret = a_tensor;
        } else {
            this->ret = b_tensor;
        }
    }

    switch (op_type) {
        case internal::SCALAR_PROD_TENSOR:
            a_tensor->get_memory_manager()->sync(true);
            internal::scalar_tensor_product_full(alpha * a_tensor->get(0), b_tensor, this->ret); break;
        case internal::TENSOR_PROD_SCALAR:
            b_tensor->get_memory_manager()->sync(true);
            internal::scalar_tensor_product_full(alpha * b_tensor->get(0), a_tensor, this->ret); break;
        case internal::TENSOR_PROD_TENSOR:
            internal::product_full(alpha, a_tensor, b_tensor, this->ret); break;
        default:
            internal::debugf("INVALID PRODUCT\n");
    }

    return this->ret;
}

template <typename T>
Operation<T> *ProductOp<T>::grad(Operation<T> *consumer, Operation<T> *var, Operation<T> *grad) {
    return NULL;
}
template class ProductOp<int>;
template class ProductOp<float>;
template class ProductOp<double>;


template <typename T>
ProductOp<T> *product(Operation<T> *a, Operation<T> *b, bool copy, bool needs_grad) {
    return product((T) 1, a, b, copy, needs_grad);
}
template ProductOp<int> *product(Operation<int> *a, Operation<int> *b, bool copy, bool needs_grad);
template ProductOp<float> *product(Operation<float> *a, Operation<float> *b, bool copy, bool needs_grad);
template ProductOp<double> *product(Operation<double> *a, Operation<double> *b, bool copy, bool needs_grad);

template <typename T>
ProductOp<T> *product(T alpha, Operation<T> *a, Operation<T> *b, bool copy, bool needs_grad) {
    return new ProductOp<T> (alpha, a, b, copy, needs_grad);
}
template ProductOp<int> *product(int alpha, Operation<int> *a, Operation<int> *b, bool copy, bool needs_grad);
template ProductOp<float> *product(float alpha, Operation<float> *a, Operation<float> *b, bool copy, bool needs_grad);
template ProductOp<double> *product(double alpha, Operation<double> *a, Operation<double> *b, bool copy, bool needs_grad);



}   // namespace op
}   // namespace magmadnn