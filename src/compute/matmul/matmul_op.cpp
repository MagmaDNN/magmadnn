/**
 * @file matmul_op.cpp
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-20
 * 
 * @copyright Copyright (c) 2019
 */
#include "compute/matmul/matmul_op.h"

namespace skepsi {
namespace op {

template <typename T>
tensor<T>* matmul_op<T>::eval() {
	tensor<T>* a_tensor = a->eval();    // MxK
	tensor<T>* b_tensor = b->eval();    // KxN
    tensor<T>* c_tensor;                // MxN

    // if a valid c operation was passed in, then use it.
    if (c != nullptr) {
        c_tensor = c->eval();
    } else {
        // TODO: fix this. don't instantiate a tensor in a gemm call. this is a performance 
        // critical function. perhaps propogate shape through operations and allocate tensor
        // before creating matmul_op.
        c_tensor = new tensor<T> ({a_tensor->get_shape(0), b_tensor->get_shape(1)}, a_tensor->get_memory_type());
        copy = false;
    }

    tensor<T>* ret;
    if (copy) {
        ret = new tensor<T> ({a_tensor->get_shape(0), b_tensor->get_shape(1)}, a_tensor->get_memory_type());
        ret->copy_from(*c_tensor);
    } else {
        ret = c_tensor;
    }

    internal::gemm_full(alpha, a_tensor, b_tensor, beta, ret);

    return ret;
} 
template class matmul_op<int>;
template class matmul_op<float>;
template class matmul_op<double>;


template <typename T>
matmul_op<T>* matmul(operation<T> *a, operation<T> *b) {
    return new matmul_op<T> ((T)1, a, b, (T)0, nullptr, false);
}
template matmul_op<int>* matmul(operation<int> *a, operation<int> *b);
template matmul_op<float>* matmul(operation<float> *a, operation<float> *b);
template matmul_op<double>* matmul(operation<double> *a, operation<double> *b);

template <typename T>
matmul_op<T>* matmul(T alpha, operation<T> *a, operation<T> *b, T beta, operation<T> *c, bool copy) {
    return new matmul_op<T> (alpha, a, b, beta, c, copy);
}
template matmul_op<int>* matmul(int alpha, operation<int> *a, operation<int> *b, int beta, operation<int> *c, bool copy);
template matmul_op<float>* matmul(float alpha, operation<float> *a, operation<float> *b, float beta, operation<float> *c, bool copy);
template matmul_op<double>* matmul(double alpha, operation<double> *a, operation<double> *b, double beta, operation<double> *c, bool copy);

} // namespace op
} // namespace skepsi
