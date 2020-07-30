#pragma once

#if defined(MAGMADNN_CMAKE_BUILD)
#include "magmadnn/config.h"
#endif
#include "magmadnn/exception.h"

namespace magmadnn {

/**
 * Marks a function as not yet implemented.
 *
 * Attempts to call this function will result in a runtime error of type
 * NotImplemented.
 */
#define MAGMADNN_NOT_IMPLEMENTED                                        \
   {                                                                    \
      throw ::magmadnn::NotImplemented(__FILE__, __LINE__, __func__);   \
   }                                                                    \
   static_assert(true,                                                  \
                 "This assert is used to counter the false positive extra " \
                 "semi-colon warnings")

   
#if defined(MAGMADNN_HAVE_MKLDNN)

/**
 * Instantiates a CudaError.
 *
 * @param errcode  The error code returned from a CUDA runtime API routine.
 */
#define MAGMADNN_DNNL_ERROR(_errcode) \
    ::magmadnn::DnnlError(__FILE__, __LINE__, __func__, _errcode)

/**
 * Asserts that a DNNL library call completed without errors.
 *
 * @param _dnnl_call  a library call expression
 */
#define MAGMADNN_ASSERT_NO_DNNL_ERRORS(_dnnl_call) \
   do {                                            \
       auto _errcode = _dnnl_call;                 \
       if (_errcode != dnnl_success) {             \
          throw MAGMADNN_DNNL_ERROR(_errcode);     \
       }                                           \
   } while (false)

#endif

/**
 * Instantiates a CudaError.
 *
 * @param errcode  The error code returned from a CUDA runtime API routine.
 */
#define MAGMADNN_CUDA_ERROR(_errcode) \
    ::magmadnn::CudaError(__FILE__, __LINE__, __func__, _errcode)


/**
 * Instantiates a CublasError.
 *
 * @param errcode  The error code returned from the cuBLAS routine.
 */
#define MAGMADNN_CUBLAS_ERROR(_errcode) \
    ::magmadnn::CublasError(__FILE__, __LINE__, __func__, _errcode)

/**
 * Instantiates a CudnnError.
 *
 * @param errcode  The error code returned from the cuBLAS routine.
 */
#define MAGMADNN_CUDNN_ERROR(_errcode) \
    ::magmadnn::CudnnError(__FILE__, __LINE__, __func__, _errcode)

/**
 * Asserts that a CUDA library call completed without errors.
 *
 * @param _cuda_call  a library call expression
 */
#define MAGMADNN_ASSERT_NO_CUDA_ERRORS(_cuda_call) \
    do {                                      \
        auto _errcode = _cuda_call;           \
        if (_errcode != cudaSuccess) {        \
            throw MAGMADNN_CUDA_ERROR(_errcode);   \
        }                                     \
    } while (false)


/**
 * Asserts that a cuBLAS library call completed without errors.
 *
 * @param _cublas_call  a library call expression
 */
#define MAGMADNN_ASSERT_NO_CUBLAS_ERRORS(_cublas_call) \
    do {                                          \
        auto _errcode = _cublas_call;             \
        if (_errcode != CUBLAS_STATUS_SUCCESS) {  \
            throw MAGMADNN_CUBLAS_ERROR(_errcode);     \
        }                                         \
    } while (false)

/**
 * Asserts that a cuBLAS library call completed without errors.
 *
 * @param _cublas_call  a library call expression
 */
#define MAGMADNN_ASSERT_NO_CUDNN_ERRORS(_cudnn_call) \
    do {                                          \
        auto _errcode = _cudnn_call;             \
        if (_errcode != CUDNN_STATUS_SUCCESS) {  \
            throw MAGMADNN_CUDNN_ERROR(_errcode);     \
        }                                         \
    } while (false)
   
} // magmadnn namespace
