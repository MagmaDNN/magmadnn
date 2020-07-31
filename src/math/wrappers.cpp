/**
 * @file wrappers.h
 * @author Florent Lopez
 * @version 0.1
 * @date 2019-12-04
 *
 * @copyright Copyright (c) 2019
 */

#include <iostream>

#include "math/wrappers.h"

extern "C" {
// GEMM
void dgemm_(char* transa, char* transb, int* m, int* n, int* k, double* alpha, const double* a, int* lda,
            const double* b, int* ldb, double* beta, double* c, int* ldc);
void sgemm_(char* transa, char* transb, int* m, int* n, int* k, float* alpha, const float* a, int* lda, const float* b,
            int* ldb, float* beta, float* c, int* ldc);
// GEMV
void sgemv_(char* trans, int* m, int* n, float* alpha, float const* a, int* lda, const float* x, int const* incx,
            float* beta, float* y, int const* incy);
void dgemv_(char* trans, int* m, int* n, double* alpha, double const* a, int* lda, const double* x, int const* incx,
            double* beta, double* y, int const* incy);
// AXPY
void daxpy_(const int* n, const double* a, const double* x, const int* incx, double* y, const int* incy);
void saxpy_(const int* n, const float* a, const float* x, const int* incx, float* y, const int* incy);
// SCAL
void dscal_(int const* n, double const* a, double const* x, int const* incx);
void sscal_(int const* n, float const* a, float const* x, int const* incx);
}

namespace magmadnn {
namespace math {

// DGEMM
template <>
void gemm<double>(enum operation transa, enum operation transb, int m, int n, int k, double alpha, const double* a,
                  int lda, const double* b, int ldb, double beta, double* c, int ldc) {
    char ftransa = (transa == OP_N) ? 'N' : 'T';
    char ftransb = (transb == OP_N) ? 'N' : 'T';
    dgemm_(&ftransa, &ftransb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}
// SGEMM
template <>
void gemm<float>(enum operation transa, enum operation transb, int m, int n, int k, float alpha, const float* a,
                 int lda, const float* b, int ldb, float beta, float* c, int ldc) {
    char ftransa = (transa == OP_N) ? 'N' : 'T';
    char ftransb = (transb == OP_N) ? 'N' : 'T';
    sgemm_(&ftransa, &ftransb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

// INT
template <>
void gemv<int>(enum operation trans, int m, int n, int alpha, int const* a, int lda, int const* x, int incx, int beta,
               int* y, int incy) {
    // char ftrans = (trans==OP_N) ? 'N' : 'T';
    // sgemv_(&ftrans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    std::cout << "gemv NOT implemented for type int" << std::endl;
}
// SGEMV
template <>
void gemv<float>(enum operation trans, int m, int n, float alpha, float const* a, int lda, float const* x, int incx,
                 float beta, float* y, int incy) {
    char ftrans = (trans == OP_N) ? 'N' : 'T';
    sgemv_(&ftrans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
}
// DGEMV
template <>
void gemv<double>(enum operation trans, int m, int n, double alpha, double const* a, int lda, double const* x, int incx,
                  double beta, double* y, int incy) {
    char ftrans = (trans == OP_N) ? 'N' : 'T';
    dgemv_(&ftrans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
}

// SAXPY
template <>
void axpy<float>(int n, float a, const float* x, int incx, float* y, int incy) {
    saxpy_(&n, &a, x, &incx, y, &incy);
}
// DAXPY
template <>
void axpy<double>(int n, double a, const double* x, int incx, double* y, int incy) {
    daxpy_(&n, &a, x, &incx, y, &incy);
}

// SSCAL
template <>
void scal<float>(int n, float a, const float* x, int incx) {
    sscal_(&n, &a, x, &incx);
}
// DSCAL
template <>
void scal<double>(int n, double a, const double* x, int incx) {
    dscal_(&n, &a, x, &incx);
}

}  // namespace math
}  // namespace magmadnn
