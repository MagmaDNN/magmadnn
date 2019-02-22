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
	tensor<T>* a_tensor = a->eval();    // MxK
	tensor<T>* b_tensor = b->eval();    // KxN

    unsigned int M = a_tensor->get_shape()[0];
    unsigned int N = b_tensor->get_shape()[1];

    tensor<T>* c_tensor = new tensor<T> ({M,N}, a_tensor->get_memory_type());

    internal::gemm_full((T) 1, a_tensor, b_tensor, (T) 0, c_tensor);

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
