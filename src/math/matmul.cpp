/**
 * @file matmul.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-06-06
 *
 * @copyright Copyright (c) 2019
 */
#include "math/matmul.h"

namespace magmadnn {
namespace math {

#define SPECIALIZE_MATMUL_INT(int_type)                                                                      \
    template <>                                                                                              \
    void matmul(int_type alpha, bool trans_A, const Tensor &A, bool trans_B, const Tensor &B, int_type beta, \
                Tensor &C) {}
CALL_FOR_ALL_INT_TYPES(SPECIALIZE_MATMUL_INT)
#undef SPECIALIZE_MATMUL_INT

template <>
void matmul(float alpha, bool trans_A, const Tensor &A, bool trans_B, const Tensor &B, float beta, Tensor &C) {
    unsigned int M, N, K, ldda, lddb, lddc;

    /* op(A) : MxK ; op(B) : KxN ; C : MxN */

    M = C.shape(0);                 /* rows of C and op(A) */
    N = C.shape(1);                 /* columns of C and op(B) */
    K = A.shape((trans_A) ? 0 : 1); /* columns of op(A) and rows of op(B) */
    ldda = (trans_A) ? M : K;       /* leading dimension of op(A) */
    lddb = (trans_B) ? K : N;       /* leading dimension of op(B) */
    lddc = N;

    // A: MxK  B: KxN  C: MxN
    // (MxR)(RxN) + (MxN) = (MxN) + (MxN) = (MxN)

    if (A.get_memory_type() == HOST) {
        CBLAS_TRANSPOSE a_trans = (trans_A) ? CblasTrans : CblasNoTrans;
        CBLAS_TRANSPOSE b_trans = (trans_B) ? CblasTrans : CblasNoTrans;

        // specify ROW MAJOR, since tensors are stored in row-major
        cblas_sgemm(CblasRowMajor, a_trans, b_trans, M, N, K, alpha, A.get_ptr<float>(), ldda, B.get_ptr<float>(), lddb,
                    beta, C.get_ptr<float>(), lddc);
    }
#if defined(_HAS_CUDA_)
    else {
        // since magma is column-major we'll need the transpose of everything
        // i.e. (AB)^T = (C)^T and the fact that (AB)^T = (B^T)(A^T)
        magma_trans_t a_trans = (trans_A) ? MagmaTrans : MagmaNoTrans;
        magma_trans_t b_trans = (trans_B) ? MagmaTrans : MagmaNoTrans;

        MAGMA_SGEMM_ROWMAJOR(A.get_ptr<float>(), B.get_ptr<float>(), C.get_ptr<float>(), M, N, K, alpha, beta, a_trans,
                             b_trans, ldda, lddb, lddc);
    }
#endif
}

template <>
void matmul(double alpha, bool trans_A, const Tensor &A, bool trans_B, const Tensor &B, double beta, Tensor &C) {
    unsigned int M, N, K, ldda, lddb, lddc;

    /* op(A) : MxK ; op(B) : KxN ; C : MxN */
    /* lddX is the leading dimension of X or the number of elements between rows.  */

    M = C.shape(0);                 /* rows of C and op(A) */
    N = C.shape(1);                 /* columns of C and op(B) */
    K = A.shape((trans_A) ? 0 : 1); /* columns of op(A) and rows of op(B) */
    ldda = (trans_A) ? M : K;       /* leading dimension of op(A) */
    lddb = (trans_B) ? K : N;       /* leading dimension of op(B) */
    lddc = N;

    // A: MxK  B: KxN  C: MxN
    // (MxR)(RxN) + (MxN) = (MxN) + (MxN) = (MxN)

    if (A.get_memory_type() == HOST) {
        CBLAS_TRANSPOSE a_trans = (trans_A) ? CblasTrans : CblasNoTrans;
        CBLAS_TRANSPOSE b_trans = (trans_B) ? CblasTrans : CblasNoTrans;

        // specify ROW MAJOR, since tensors are stored in row-major
        cblas_dgemm(CblasRowMajor, a_trans, b_trans, M, N, K, alpha, A.get_ptr<double>(), ldda, B.get_ptr<double>(),
                    lddb, beta, C.get_ptr<double>(), lddc);
    }
#if defined(_HAS_CUDA_)
    else {
        // since magma is column-major we'll need the transpose of everything
        // i.e. (AB)^T = (C)^T and the fact that (AB)^T = (B^T)(A^T)
        magma_trans_t a_trans = (trans_A) ? MagmaTrans : MagmaNoTrans;
        magma_trans_t b_trans = (trans_B) ? MagmaTrans : MagmaNoTrans;

        MAGMA_DGEMM_ROWMAJOR(A.get_ptr<double>(), B.get_ptr<double>(), C.get_ptr<double>(), M, N, K, alpha, beta,
                             a_trans, b_trans, ldda, lddb, lddc);
    }
#endif
}

}  // namespace math
}  // namespace magmadnn