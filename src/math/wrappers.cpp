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
   void dgemm_(char* transa, char* transb, int* m, int* n, int* k, double* alpha, const double* a, int* lda, const double* b, int* ldb, double *beta, double* c, int* ldc);
   void sgemm_(char* transa, char* transb, int* m, int* n, int* k, float* alpha, const float* a, int* lda, const float* b, int* ldb, float *beta, float* c, int* ldc);
   // GEMV
   void sgemv_(char* trans, int *m, int *n, float* alpha, float const* a, int* lda, const float *x, int const* incx, float *beta, float *y, int const *incy);
   void dgemv_(char* trans, int *m, int *n, double* alpha, double const* a, int* lda, const double *x, int const* incx, double *beta, double *y, int const *incy);
}

namespace magmadnn {
namespace math {

   // DGEMM
   template <>
   void gemm<double>(
         enum operation transa, enum operation transb,
         int m, int n, int k, double alpha, const double* a, int lda,
         const double* b, int ldb, double beta, double* c, int ldc) {
      char ftransa = (transa==OP_N) ? 'N' : 'T';
      char ftransb = (transb==OP_N) ? 'N' : 'T';
      dgemm_(&ftransa, &ftransb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
   }
   // SGEMM
   template <>
   void gemm<float>(
         enum operation transa, enum operation transb,
         int m, int n, int k, float alpha, const float * a, int lda,
         const float * b, int ldb, float beta, float* c, int ldc) {
      char ftransa = (transa==OP_N) ? 'N' : 'T';
      char ftransb = (transb==OP_N) ? 'N' : 'T';
      sgemm_(&ftransa, &ftransb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
   }

   // INT
   template<>
   void gemv<int>(enum operation trans, int m, int n, int alpha, int const* a, int lda,
                    int const* x, int incx, int beta, int *y, int incy) {
      // char ftrans = (trans==OP_N) ? 'N' : 'T';
      // sgemv_(&ftrans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
      std::cout << "gemv NOT implemented for type int" << std::endl;
   }   
   // SGEMV
   template<>
   void gemv<float>(enum operation trans, int m, int n, float alpha, float const* a, int lda,
                    float const* x, int incx, float beta, float *y, int incy) {
      char ftrans = (trans==OP_N) ? 'N' : 'T';
      sgemv_(&ftrans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
   }
   // DGEMV
   template<>
   void gemv<double>(enum operation trans, int m, int n, double alpha, double const* a, int lda,
                     double const* x, int incx, double beta, double *y, int incy) {
      char ftrans = (trans==OP_N) ? 'N' : 'T';
      dgemv_(&ftrans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
   }

   
}}  // namespace magmadnn::math
