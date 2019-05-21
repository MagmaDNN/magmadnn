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
ProductOp<T>::ProductOp(T alpha, Operation<T> *a, Operation<T> *b, bool copy) {
    
    if (copy) {
        this->ret = new Tensor<T> (a->get_output_shape(), {ONE, {}}, a->get_memory_type());
    }
}

template <typename T>
Tensor<T> *ProductOp<T>::eval() {
    a_tensor = a->eval();
    b_tensor = b->eval();
    
    if (!copy) {
        this->ret = b_tensor;
    }

    internal::product_full(alpha, a_tensor, b_tensor, this->ret);

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
ProductOp<T> *product(Operation<T> *a, Operation<T> *b, bool copy) {
    return product((T) 1, a, b, copy);
}
template ProductOp<int> *product(Operation<int> *a, Operation<int> *b, bool copy);
template ProductOp<float> *product(Operation<float> *a, Operation<float> *b, bool copy);
template ProductOp<double> *product(Operation<double> *a, Operation<double> *b, bool copy);

template <typename T>
ProductOp<T> *product(T alpha, Operation<T> *a, Operation<T> *b, bool copy) {
    return new ProductOp<T> (alpha, a, b, copy);
}
template ProductOp<int> *product(int alpha, Operation<int> *a, Operation<int> *b, bool copy);
template ProductOp<float> *product(float alpha, Operation<float> *a, Operation<float> *b, bool copy);
template ProductOp<double> *product(double alpha, Operation<double> *a, Operation<double> *b, bool copy);



}   // namespace op
}   // namespace magmadnn