/**
 * @file matmul.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-06-06
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "tensor/tensor.h"
#include "cblas.h"
#if defined(_HAS_CUDA_)
#include "magma.h"
#endif


#define MAGMA_SGEMM_ROWMAJOR(A,B,C,m,n,k,alpha,beta,transf_A,transf_B,lda,ldb,ldc) \
    magma_sgemm(transf_B, transf_A, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc)

#define MAGMA_DGEMM_ROWMAJOR(A,B,C,m,n,k,alpha,beta,transf_A,transf_B,lda,ldb,ldc) \
    magma_dgemm(transf_B, transf_A, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc)


namespace magmadnn {
namespace math {

template <typename T>
void matmul(T alpha, bool trans_A, Tensor<T> *A, bool trans_B, Tensor<T> *B, T beta, Tensor<T> *C);

}
}