/**
 * @file dot.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-06-07
 * 
 * @copyright Copyright (c) 2019
 */
#include "math/dot.h"

namespace magmadnn {
namespace math {

template <typename T>
void dot(Tensor<T> *A, Tensor<T> *B, Tensor<T> *out) {
    dot(1.0, A, B, 0.0, out);
}

template <typename T>
void dot(T alpha, Tensor<T> *A, Tensor<T> *B, T beta, Tensor<T> *out) {
    dot(alpha, false, A, false, B, beta, out);
}

template <typename T>
void dot(T alpha, bool trans_A, Tensor<T> *A, bool trans_B, Tensor<T> *B, T beta, Tensor<T> *out) {

    unsigned int n_axes_a = A->get_shape().size();
    unsigned int n_axes_b = B->get_shape().size();
    bool a_is_scalar = (A->get_size() == 1);
    bool b_is_scalar = (B->get_size() == 1);

    if (n_axes_a == 2 && n_axes_b == 2) {
        /* matmul(A,B) */

        matmul(alpha, trans_A, A, trans_B, B, beta, out);

    } else if (n_axes_a == 2 && n_axes_b == 1 && !b_is_scalar) {
        /* gemv(A,B) */
        fprintf(stderr, "add gemv\n");
    } else if (n_axes_a == 1 && n_axes_b == 2 && !a_is_scalar) {
        /* gemv(B^T,A) */
        fprintf(stderr, "add gemv\n");
    } else if (a_is_scalar || b_is_scalar) {
        /* broadcast product */
        fprintf(stderr, "add prod\n");

        

    } else {
        /* other */
        fprintf(stderr, "undefined dot product!");
    }
}
template void dot(int alpha, bool trans_A, Tensor<int> *A, bool trans_B, Tensor<int> *B, int beta, Tensor<int> *out);
template void dot(float alpha, bool trans_A, Tensor<float> *A, bool trans_B, Tensor<float> *B, float beta, Tensor<float> *out);
template void dot(double alpha, bool trans_A, Tensor<double> *A, bool trans_B, Tensor<double> *B, double beta, Tensor<double> *out);


}
}