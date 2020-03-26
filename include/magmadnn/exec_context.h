#pragma once

#include "magmadnn/exception_helpers.h"   

#include <iostream>

#if defined(MAGMADNN_HAVE_CUDA)
#include <cuda.h>   
#include <cublas_v2.h>
#include <cudnn.h>
#endif


namespace magmadnn {

class ExecContext {
public:
   ExecContext()
      : id_(-1)
   {}

   int id() const { return id_; }

   void id(int in_id) {
      id_ = in_id;
   }

private:
   // Worker index
   int id_;

};
   
#if defined(MAGMADNN_HAVE_CUDA)

class CudaExecContext : public ExecContext {
public:

   CudaExecContext()
      : devid_(0), custream_(nullptr), cublas_handle_(nullptr),
        cudnn_handle_(nullptr)
   {

      cudaError_t cuerr;
      cudnnStatus_t cudnnstat;
      cublasStatus_t cublasstat;

      // std::cout << "[CudaExecContext]" << std::endl;
      
      // Create CUDA Stream
      cuerr = cudaStreamCreate(&custream_);
      MAGMADNN_ASSERT_NO_CUDA_ERRORS(cuerr);

      // Create cuDNN handle and associate CUDA stream
      cudnnstat = cudnnCreate(&cudnn_handle_);
      MAGMADNN_ASSERT_NO_CUDNN_ERRORS(cudnnstat);
      cudnnstat = cudnnSetStream(cudnn_handle_, custream_);
      MAGMADNN_ASSERT_NO_CUDNN_ERRORS(cudnnstat);

      // Create cuBLAS handle and associate CUDA stream
      cublasstat = cublasCreate(&cublas_handle_);
      MAGMADNN_ASSERT_NO_CUBLAS_ERRORS(cublasstat);
      cublasstat = cublasSetStream(cublas_handle_, custream_);     
      MAGMADNN_ASSERT_NO_CUBLAS_ERRORS(cublasstat);

   }

   ~CudaExecContext() {
      // std::cout << "[~CudaExecContext]" << std::endl;

      cudaStreamDestroy(custream_);
      cudnnDestroy(cudnn_handle_);
      cublasDestroy(cublas_handle_);
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

   void synchronize() const {
      cudaError_t cuerr;

      cuerr = cudaStreamSynchronize(custream_);
      MAGMADNN_ASSERT_NO_CUDA_ERRORS(cuerr);

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
