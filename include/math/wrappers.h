/**
 * @file wrappers.h
 * @author Florent Lopez
 * @version 0.1
 * @date 2019-12-04
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

namespace magmadnn {
namespace math {

   /// @brief magmadnn::math::operation enumerates operations that can
   /// be applied to a matrix * argument of a BLAS call.
   enum operation
      {
       /// No operation (i.e. non-transpose). Equivalent to BLAS op='N'.
       OP_N,
       /// Transposed. Equivalent to BLAS op='T'.
       OP_T
      };

   /* _GEMM */
   template <typename T>
   void gemm(enum operation transa,
             enum operation transb,
             int m, int n, int k, T alpha, const T* a, int lda, const T* b,
             int ldb, T beta, T* c, int ldc);

   // GEMV
   template <typename T>
   void gemv(enum operation trans, int m, int n, T alpha, T const* a, int lda,
                  T const* x, int incx, T beta, T *y, int incy);

   // AXPY
   template <typename T> 
   void axpy(int n, const T a, const T *x, const int incx, T *y, const int incy);

   // SCAL
   template <typename T> 
   void scal(int n, const T a, const T *x, const int incx);

}}  // namespace magmadnn::math
