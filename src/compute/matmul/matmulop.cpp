/**
 * @file matmulop.cpp
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-02-20
 * 
 * @copyright Copyright (c) 2019
 */
#include "compute/matmul/matmulop.h"

namespace magmadnn {
namespace op {

template <typename T>
MatmulOp<T>::MatmulOp(T alpha, Operation<T>* a, Operation<T>* b, T beta, Operation<T> *c, bool copy, bool needs_grad) : 
		Operation<T>::Operation({a,b,c}, needs_grad), a(a), b(b), c(c), alpha(alpha), beta(beta), copy(copy) {

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
        this->output_tensor = new Tensor<T> (this->output_shape, this->mem_type);
    }

    /* init gradient tensors to NULL */
    this->_grad_cache[(uintptr_t)a] = NULL;
    this->_grad_cache[(uintptr_t)b] = NULL;
    this->_grad_cache[(uintptr_t)c] = NULL;
}

template <typename T>
Tensor<T>* MatmulOp<T>::_eval(bool recompute) {

	a_tensor = a->eval(recompute);    // MxK
	b_tensor = b->eval(recompute);    // KxN
    c_tensor = c->eval(recompute);

    if (copy) {
        this->output_tensor->copy_from(*c_tensor);
    } else {
        this->output_tensor = c_tensor;
    }

    internal::gemm_full(alpha, a_tensor, b_tensor, beta, this->output_tensor);

    return this->output_tensor;
} 

template <typename T>
Tensor<T> *MatmulOp<T>::_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad) {
    /* wrt a: dot(grad, B^T)  |  wrt b: dot(a^T, grad) */
    Tensor<T> *out = this->_grad_cache[(uintptr_t)var];

    if (var == a) {
        b_tensor = b->eval(false);  /* don't recalculate if necessary */

        /* init grad tensor */
        if (out == NULL) {
            if (T_IS_MATRIX(b_tensor)) {
                /* grad.B^T has shape row(grad) x row(B) */
                out = new Tensor<T> ({grad->get_shape(0), b_tensor->get_shape(0)}, {NONE,{}}, this->mem_type);
            } else if (T_IS_VECTOR(a_tensor)) {
                /* grad.B^T has shape row(B) */
                out = new Tensor<T> ({b_tensor->get_shape(0)}, {NONE,{}}, this->mem_type);
            } else {
                /* grad.B^T has shape of B^T */
                out = new Tensor<T> ({b_tensor->get_shape(1), b_tensor->get_shape(0)}, {NONE,{}}, this->mem_type);
            }
            
            this->_grad_cache[(uintptr_t)a] = out;
        }

        math::dot((T)1, false, grad, true, b_tensor, (T)0, out);

        //return dot(grad, transpose(b, true, false), true, false);
    } else {
        a_tensor = a->eval(false);

        /* need to create grad out for this */
        if (out == NULL) {
            if (T_IS_MATRIX(grad)) {
                /* if grad is a matrix, then a^T grad is the gradient */
                out = new Tensor<T> ({a_tensor->get_shape(1), grad->get_shape(1)}, {NONE,{}}, this->mem_type);
            } else if (T_IS_VECTOR(grad)) {
                /* if grad is a vector, then a^T grad has the same shape as col(a) */
                out = new Tensor<T> ({a_tensor->get_shape(1)}, {NONE,{}}, this->mem_type);
            } else {
                /* if grad is a scalar?, then a^T grad has the same shape as A^T */
                out = new Tensor<T> ({a_tensor->get_shape(1), a_tensor->get_shape(0)}, {NONE,{}}, this->mem_type);
            }
            this->_grad_cache[(uintptr_t)b] = out;
        }

        math::dot((T)1, true, a_tensor, false, grad, (T)0, out);

        //return dot(transpose(a, true, false), grad, true, false);
    }
    return out;
}
template class MatmulOp<int>;
template class MatmulOp<float>;
template class MatmulOp<double>;


template <typename T>
MatmulOp<T>* matmul(Operation<T> *a, Operation<T> *b, bool needs_grad) {
    
    assert( a->get_output_shape().size() == 2 );
    assert( b->get_output_shape().size() == 2 );

    Tensor<T> *c_tensor = new Tensor<T> ({a->get_output_shape(0), b->get_output_shape(1)}, a->get_memory_type());
    Operation<T> *c = var("__matmul_c_internal", c_tensor);
    return new MatmulOp<T> ((T)1, a, b, (T)0, c, false, needs_grad);
}
template MatmulOp<int>* matmul(Operation<int> *a, Operation<int> *b, bool needs_grad);
template MatmulOp<float>* matmul(Operation<float> *a, Operation<float> *b, bool needs_grad);
template MatmulOp<double>* matmul(Operation<double> *a, Operation<double> *b, bool needs_grad);

template <typename T>
MatmulOp<T>* matmul(T alpha, Operation<T> *a, Operation<T> *b, T beta, Operation<T> *c, bool copy, bool needs_grad) {
    return new MatmulOp<T> (alpha, a, b, beta, c, copy);
}
template MatmulOp<int>* matmul(int alpha, Operation<int> *a, Operation<int> *b, int beta, Operation<int> *c, bool copy, bool needs_grad);
template MatmulOp<float>* matmul(float alpha, Operation<float> *a, Operation<float> *b, float beta, Operation<float> *c, bool copy, bool needs_grad);
template MatmulOp<double>* matmul(double alpha, Operation<double> *a, Operation<double> *b, double beta, Operation<double> *c, bool copy, bool needs_grad);

} // namespace op
} // namespace magmadnn
