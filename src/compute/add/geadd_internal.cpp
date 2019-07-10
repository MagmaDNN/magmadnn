/**
 * @file geadd_internal.cpp
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-02-22
 *
 * @copyright Copyright (c) 2019
 */
#include "compute/add/geadd_internal.h"

namespace magmadnn {
namespace internal {

template <typename T>
bool geadd_check(Tensor<T> *A, Tensor<T> *B, Tensor<T> *C) {
    assert(A->get_shape().size() == 2);
    assert(B->get_shape().size() == 2);
    assert(C->get_shape().size() == 2);

    assert(A->get_shape(0) == B->get_shape(0));
    assert(A->get_shape(0) == C->get_shape(0));
    assert(A->get_shape(1) == B->get_shape(1));
    assert(A->get_shape(1) == C->get_shape(1));
    return true;
}

template <typename T>
void geadd_full(T alpha, Tensor<T> *A, T beta, Tensor<T> *B, Tensor<T> *C) {
    if (A->get_memory_type() == HOST) {
        T *a_ptr = A->get_ptr();
        T *b_ptr = B->get_ptr();
        T *c_ptr = C->get_ptr();
        unsigned int size = A->get_size();

        for (unsigned int i = 0; i < size; i++) {
            c_ptr[i] = (alpha * a_ptr[i]) + (beta * b_ptr[i]);
        }
    }
#if defined(_HAS_CUDA_)
    else {
        geadd_full_device(alpha, A, beta, B, C);
    }
#endif
}
template void geadd_full(int alpha, Tensor<int> *A, int beta, Tensor<int> *B, Tensor<int> *C);
template void geadd_full(float alpha, Tensor<float> *A, float beta, Tensor<float> *B, Tensor<float> *C);
template void geadd_full(double alpha, Tensor<double> *A, double beta, Tensor<double> *B, Tensor<double> *C);

template <typename T>
void tensor_scalar_add_full(T alpha, Tensor<T> *x, Tensor<T> *out) {
    if (out->get_memory_type() == HOST) {
        T *x_ptr = x->get_ptr();
        T *out_ptr = out->get_ptr();
        unsigned int size = out->get_size();

        for (unsigned int i = 0; i < size; i++) {
            out_ptr[i] = alpha + x_ptr[i];
        }
    }
#if defined(_HAS_CUDA_)
    else {
        tensor_scalar_add_full_device(alpha, x, out);
    }
#endif
}
template void tensor_scalar_add_full(int alpha, Tensor<int> *x, Tensor<int> *out);
template void tensor_scalar_add_full(float alpha, Tensor<float> *x, Tensor<float> *out);
template void tensor_scalar_add_full(double alpha, Tensor<double> *x, Tensor<double> *out);

}  // namespace internal
}  // namespace magmadnn
