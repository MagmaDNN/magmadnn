/**
 * @file wrappers.h
 * @author Florent Lopez
 * @version 0.1
 * @date 2019-12-04
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "math/wrappers.h"

extern "C" {
   // GEMM
   void dgemm_(char* transa, char* transb, int* m, int* n, int* k, double* alpha, const double* a, int* lda, const double* b, int* ldb, double *beta, double* c, int* ldc);
   void sgemm_(char* transa, char* transb, int* m, int* n, int* k, float* alpha, const float* a, int* lda, const float* b, int* ldb, float *beta, float* c, int* ldc);

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

   
}}  // namespace magmadnn::math
