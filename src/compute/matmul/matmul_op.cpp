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
matmul_op<T>::matmul_op(T alpha, operation<T>* a, operation<T>* b, T beta, operation<T> *c, bool copy) : 
		operation<T>::operation({a,b,c}), a(a), b(b), c(c), alpha(alpha), beta(beta), copy(copy) {

    unsigned int M, N, K;

    // must have same memory types
	assert( a->get_memory_type() == b->get_memory_type() );
	assert( b->get_memory_type() == c->get_memory_type() );

	// tensors must be matrices
	assert( a->get_output_shape().size() == 2 );
	assert( b->get_output_shape().size() == 2 );
	assert( c->get_output_shape().size() == 2 );

	// A: MxK  B: KxN  C: MxN
	M = a->get_output_shape(0);
	K = a->get_output_shape(1);
	N = b->get_output_shape(1);

	// valid shapes
	assert( b->get_output_shape(0) == K );
	assert( c->get_output_shape(0) == M );
	assert( c->get_output_shape(1) == N );

    this->output_shape = {M,N};
    this->mem_type = a->get_memory_type();

    /* avoid allocating memory in eval */
    if (copy) {
        ret = new tensor<T> (this->output_shape, this->mem_type);
    }
}

template <typename T>
tensor<T>* matmul_op<T>::eval() {
	a_tensor = a->eval();    // MxK
	b_tensor = b->eval();    // KxN
    c_tensor = c->eval();

    if (copy) {
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
    tensor<T> *c_tensor = new tensor<T> ({a->get_output_shape(0), b->get_output_shape(1)}, a->get_memory_type());
    operation<T> *c = var("__matmul_internal_c", c_tensor);
    return new matmul_op<T> ((T)1, a, b, (T)0, c, false);
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
