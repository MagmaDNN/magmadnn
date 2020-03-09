#pragma once

#include "magmadnn/exception_helpers.h"   

#if defined(MAGMADNN_HAVE_CUDA)
#include <cuda.h>   
#include <cublas_v2.h>
#include <cudnn.h>
#endif


namespace magmadnn {

#if defined(MAGMADNN_HAVE_CUDA)

class CudaExecContext {
public:

   CudaExecContext()
      : devid_(0), custream_(nullptr), cublas_handle_(nullptr),
        cudnn_handle_(nullptr)
   {

      cudaError_t cuerr;
          
      // Create Stream
      cuerr = cudaStreamCreate(&custream_);
      MAGMADNN_ASSERT_NO_CUDA_ERRORS(cuerr);

      
   }
   
   int devid() const { return devid_; }

   void devid(int in_devid) {
      devid_ = in_devid;
   }
   
   cublasHandle_t cublas_handle() const { return cublas_handle_; }

   void cublas_handle(cublasHandle_t in_cublas_handle) {
      cublas_handle_ = in_cublas_handle;
   }
   
   cudnnHandle_t cudnn_handle() const { return cudnn_handle_; }

   void cudnn_handle(cudnnHandle_t in_cudnn_handle) {
      cudnn_handle_ = in_cudnn_handle;
   }

   cudaStream_t stream() const { return custream_; }

   void stream(cudaStream_t in_stream) {
      custream_ = in_stream;
   }

private:

   // Device index
   int devid_;
   // CUDA stream
   cudaStream_t custream_;
   // CuDNN handle 
   cublasHandle_t cublas_handle_;
   // CuBLAS handle
   cudnnHandle_t cudnn_handle_;
};

#endif
   
} // End of magmadnn namespace
