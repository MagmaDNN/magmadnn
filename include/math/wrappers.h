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
   
}}  // namespace magmadnn::math
