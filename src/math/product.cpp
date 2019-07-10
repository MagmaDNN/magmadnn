/**
 * @file product.cpp
 * @author Sedrick Keh
 * @version 0.1
 * @date 2019-07-02
 *
 * @copyright Copyright (c) 2019
 */
#include "math/product.h"

namespace magmadnn {
namespace math {

template <typename T>
void product(Tensor<T> *A, Tensor<T> *B) {
    assert(A->get_size() == B->get_size());
    assert(T_IS_SAME_MEMORY_TYPE(A, B));

    T *A_ptr = A->get_ptr();
    T *B_ptr = B->get_ptr();
    unsigned int size = B->get_size();

    for (unsigned int i = 0; i < size; i++) {
        B_ptr[i] = B_ptr[i] * A_ptr[i];
    }
}
template void product(Tensor<int> *A, Tensor<int> *B);
template void product(Tensor<float> *A, Tensor<float> *B);
template void product(Tensor<double> *A, Tensor<double> *B);

template <typename T>
void product(Tensor<T> *A, Tensor<T> *B, Tensor<T> *C) {
    assert(A->get_size() == B->get_size());
    assert(T_IS_SAME_MEMORY_TYPE(A, B));
    assert(A->get_size() == C->get_size());
    assert(T_IS_SAME_MEMORY_TYPE(A, C));

    T *A_ptr = A->get_ptr();
    T *B_ptr = B->get_ptr();
    T *C_ptr = C->get_ptr();
    unsigned int size = C->get_size();

    for (unsigned int i = 0; i < size; i++) {
        C_ptr[i] = B_ptr[i] * A_ptr[i];
    }
}
template void product(Tensor<int> *A, Tensor<int> *B, Tensor<int> *C);
template void product(Tensor<float> *A, Tensor<float> *B, Tensor<float> *C);
template void product(Tensor<double> *A, Tensor<double> *B, Tensor<double> *C);

}  // namespace math
}  // namespace magmadnn