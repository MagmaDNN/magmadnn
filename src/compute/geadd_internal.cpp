/**
 * @file geadd_internal.cpp
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-22
 * 
 * @copyright Copyright (c) 2019
 */
#include "compute/geadd_internal.h"

namespace skepsi {
namespace internal {

template <typename T>
bool geadd_check(tensor<T> *A, tensor<T> *B, tensor<T> *C) {
    assert( A->get_shape().size() == 2 );
    assert( B->get_shape().size() == 2 );
    assert( C->get_shape().size() == 2 );

    assert( A->get_shape(0) == B->get_shape(0) );
    assert( A->get_shape(0) == C->get_shape(0) );
    assert( A->get_shape(1) == B->get_shape(1) );
    assert( A->get_shape(1) == C->get_shape(1) );
    return true;
}

template <typename T>
void geadd_full(T alpha, tensor<T> *A, T beta, tensor<T> *B, tensor<T> *C) {

    if (!geadd_check(A, B, C)) return;

    if (A->get_memory_type() == HOST) {
        T *a_ptr = A->get_memory_manager()->get_host_ptr();
        T *b_ptr = B->get_memory_manager()->get_host_ptr();
        T *c_ptr = C->get_memory_manager()->get_host_ptr();
        for (unsigned int i = 0; i < A->get_size(); i++) {
            c_ptr[i] = (alpha * a_ptr[i]) + (beta * b_ptr[i]);
        }
    }
    #if defined(_HAS_CUDA_)
    else {
        // geadd_full_device(alpha, A->get_memory_manager()->get_ptr(), beta, B->get_memory_manager()->get_ptr());
    }
    #endif
}

template void geadd_full(int alpha, tensor<int> *A, int beta, tensor<int> *B, tensor<int> *C);
template void geadd_full(float alpha, tensor<float> *A, float beta, tensor<float> *B, tensor<float> *C);
template void geadd_full(double alpha, tensor<double> *A, double beta, tensor<double> *B, tensor<double> *C);

}   // namespace internal
}   // namespace skepsi