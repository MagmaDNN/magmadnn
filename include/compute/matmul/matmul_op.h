/**
 * @file matmul_op.h
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-20
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once
#include <vector>
#include "compute/operation.h"
#include "compute/variable.h"
#include "tensor/tensor.h"
#include "gemm_internal.h"

namespace skepsi {
namespace op {

template <typename T>
class matmul_op : public operation<T> {
public:
	matmul_op(T alpha, operation<T>* a, operation<T>* b, T beta, operation<T> *c, bool copy=true);

	tensor<T>* eval();
	
	std::string to_string() { return "(" + a->to_string() + " * " + b->to_string() + ")"; }
protected:
	operation<T> *a;
	operation<T> *b;
	operation<T> *c;

	tensor<T> *a_tensor;
	tensor<T> *b_tensor;
	tensor<T> *c_tensor;
	tensor<T> *ret;

	T alpha;
	T beta;
	bool copy;
};

/** Returns a new operation of type matmul. It computes the matrix product of A and B.
 * @tparam T 
 * @param a 
 * @param b 
 * @param copy 
 * @return matmul_op<T>* 
 */
template <typename T>
matmul_op<T>* matmul(operation<T> *a, operation<T> *b);

/** Computes the full gemm C = alpha*(AB) + beta*(C). Overwrites C and returns it if copy is false. If true,
 * 	then it returns a copy of C.
 * @tparam T 
 * @param alpha 
 * @param a 
 * @param b 
 * @param beta 
 * @param c 
 * @param copy 
 * @return matmul_op<T>* 
 */
template <typename T>
matmul_op<T>* matmul(T alpha, operation<T> *a, operation<T> *b, T beta, tensor<T> *c, bool copy);

} // namespace op
} // namespace skepsi
