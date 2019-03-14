/**
 * @file matmulop.cpp
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-20
 * 
 * @copyright Copyright (c) 2019
 */
#include "compute/matmul/matmulop.h"

namespace skepsi {
namespace op {

template <typename T>
MatmulOp<T>::MatmulOp(T alpha, Operation<T>* a, Operation<T>* b, T beta, Operation<T> *c, bool copy) : 
		Operation<T>::Operation({a,b,c}), a(a), b(b), c(c), alpha(alpha), beta(beta), copy(copy) {

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
        ret = new Tensor<T> (this->output_shape, this->mem_type);
    }
}

template <typename T>
Tensor<T>* MatmulOp<T>::eval() {
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
template class MatmulOp<int>;
template class MatmulOp<float>;
template class MatmulOp<double>;


template <typename T>
MatmulOp<T>* matmul(Operation<T> *a, Operation<T> *b) {
    Tensor<T> *c_tensor = new Tensor<T> ({a->get_output_shape(0), b->get_output_shape(1)}, a->get_memory_type());
    Operation<T> *c = var("__matmul_internal_c", c_tensor);
    return new MatmulOp<T> ((T)1, a, b, (T)0, c, false);
}
template MatmulOp<int>* matmul(Operation<int> *a, Operation<int> *b);
template MatmulOp<float>* matmul(Operation<float> *a, Operation<float> *b);
template MatmulOp<double>* matmul(Operation<double> *a, Operation<double> *b);

template <typename T>
MatmulOp<T>* matmul(T alpha, Operation<T> *a, Operation<T> *b, T beta, Operation<T> *c, bool copy) {
    return new MatmulOp<T> (alpha, a, b, beta, c, copy);
}
template MatmulOp<int>* matmul(int alpha, Operation<int> *a, Operation<int> *b, int beta, Operation<int> *c, bool copy);
template MatmulOp<float>* matmul(float alpha, Operation<float> *a, Operation<float> *b, float beta, Operation<float> *c, bool copy);
template MatmulOp<double>* matmul(double alpha, Operation<double> *a, Operation<double> *b, double beta, Operation<double> *c, bool copy);

} // namespace op
} // namespace skepsi
