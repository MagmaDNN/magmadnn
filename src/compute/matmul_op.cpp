/**
 * @file matmul_op.cpp
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-20
 * 
 * @copyright Copyright (c) 2019
 */
#include "compute/matmul_op.h"

namespace skepsi {
namespace op {

template <typename T>
tensor<T>* matmul_op<T>::eval() {
	tensor<T>* a_tensor = a->eval();    // NxM
	tensor<T>* b_tensor = b->eval();    // MxP

    unsigned int n = a_tensor->get_shape()[0];
    unsigned int m = a_tensor->get_shape()[1];
    unsigned int p = b_tensor->get_shape()[1];
    assert( b_tensor->get_shape()[0] == m );

    tensor<T>* c_tensor = new tensor<T> ({n,p}, a_tensor->get_memory_type());

	for (int i = 0; i < (int)n; i++) {
        for (int j = 0; j < (int)p; j++) {
            T sum = (T) 0;
            for (int k = 0; k < (int)m; k++) {
                sum = sum + (a_tensor->get({i,k}) * b_tensor->get({k,j}));
            }
            c_tensor->set({i,j}, sum);
        }
    }
	return c_tensor;
} 
template class matmul_op<int>;
template class matmul_op<float>;
template class matmul_op<double>;


template <typename T>
matmul_op<T>* matmul(operation<T> *a, operation<T> *b, bool copy) {
    return new matmul_op<T> (a, b, copy);
}
template matmul_op<int>* matmul(operation<int> *a, operation<int> *b, bool copy);
template matmul_op<float>* matmul(operation<float> *a, operation<float> *b, bool copy);
template matmul_op<double>* matmul(operation<double> *a, operation<double> *b, bool copy);

} // namespace op
} // namespace skepsi
