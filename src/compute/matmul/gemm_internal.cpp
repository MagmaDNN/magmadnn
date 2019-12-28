/**
 * @file gemm_internal.cpp
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-02-22
 *
 * @copyright Copyright (c) 2019
 */
#include <cassert>

#include "math/wrappers.h"
#include "compute/matmul/gemm_internal.h"

namespace magmadnn {
namespace internal {

template <typename T>
bool gemm_check(Tensor<T> *A, Tensor<T> *B, Tensor<T> *C, /*unsigned*/ int &M, /*unsigned*/ int &N, /*unsigned*/ int &K) {
    // must have same memory types
    assert(A->get_memory_type() == B->get_memory_type());
    assert(B->get_memory_type() == C->get_memory_type());

    // tensors must be matrices
    assert(A->get_shape().size() == 2);
    assert(B->get_shape().size() == 2);
    assert(C->get_shape().size() == 2);

    // A: MxK  B: KxN  C: MxN
    M = A->get_shape(0);
    K = A->get_shape(1);
    N = B->get_shape(1);

    // valid shapes
    assert(B->get_shape(0) == K);
    assert(C->get_shape(0) == M);
    assert(C->get_shape(1) == N);

    return true;
}

/* INT */
template <>
void gemm_full(int alpha, Tensor<int> *A, Tensor<int> *B, int beta, Tensor<int> *C) {
    // unsigned int M, N, K;
    int M, N, K;
    if (!gemm_check(A, B, C, M, N, K)) return;

    // standard O(MNK) gemm algorithm
    // TODO: replace with strassen
    for (int i = 0; i < (int) M; i++) {
        for (int j = 0; j < (int) N; j++) {
            int sum = 0;
            for (int k = 0; k < (int) K; k++) {
                sum = sum + alpha * (A->get({i, k}) * B->get({k, j}));
            }
            C->set({i, j}, sum + beta * C->get({i, j}));
        }
    }
}

/* FLOAT */
template <>
void gemm_full(float alpha, Tensor<float> *A, Tensor<float> *B, float beta, Tensor<float> *C) {
    // unsigned int M, N, K;
    int M, N, K;
    if (!gemm_check(A, B, C, M, N, K)) return;

    // A: MxK  B: KxN  C: MxN
    // (MxR)(RxN) + (MxN) = (MxN) + (MxN) = (MxN)

    if (A->get_memory_type() == HOST) {
        // specify ROW MAJOR, since tensors are stored in row-major
        // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A->get_ptr(), K, B->get_ptr(), N, beta,
        //             C->get_ptr(), N);

        int lda = K;
        int ldb = N;
        int ldc = N;
        
        gemm(magmadnn::math::OP_N, magmadnn::math::OP_N,
             N, M, K, alpha, B->get_ptr(), ldb, A->get_ptr(), lda, beta, C->get_ptr(), ldc);

    }
#if defined(_HAS_CUDA_)
    else {
        // since magma is column-major we'll need the transpose of everything
        // i.e. (AB)^T = (C)^T and the fact that (AB)^T = (B^T)(A^T)
        magma_sgemm(MagmaNoTrans, MagmaNoTrans, N, M, K, alpha, B->get_ptr(), N, A->get_ptr(), K, beta, C->get_ptr(),
                    N);
    }
#endif
}

/* DOUBLE */
template <>
void gemm_full(double alpha, Tensor<double> *A, Tensor<double> *B, double beta, Tensor<double> *C) {
    // unsigned int M, N, K;
   int M, N, K;
    if (!gemm_check(A, B, C, M, N, K)) return;

    if (A->get_memory_type() == HOST) {
        // specify ROW MAJOR, since tensors are stored in row-major
        // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A->get_ptr(), K, B->get_ptr(), N, beta,
        //             C->get_ptr(), N);

        int lda = K;
        int ldb = N;
        int ldc = N;
        
        gemm(magmadnn::math::OP_N, magmadnn::math::OP_N,
             N, M, K, alpha, B->get_ptr(), ldb, A->get_ptr(), lda, beta, C->get_ptr(), ldc);

    }
#if defined(_HAS_CUDA_)
    else {
        // since magma is column-major we'll need the transpose of everything
        // i.e. (AB)^T = (C)^T and the fact that (AB)^T = (B^T)(A^T)
        magma_dgemm(MagmaNoTrans, MagmaNoTrans, N, M, K, alpha, B->get_ptr(), N, A->get_ptr(), K, beta, C->get_ptr(),
                    N);
    }
#endif
}

}  // namespace internal
}  // namespace magmadnn
