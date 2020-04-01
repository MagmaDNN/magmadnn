#include "compute/pooling/poolingop.h"

#include "magmadnn/config.h"

#include <iostream>

namespace magmadnn {
namespace op {

template <typename T>
PoolingOp<T>::PoolingOp(
      Operation<T> *input,
      int filter_h, int filter_w,
      int pad_h, int pad_w,
      int vertical_stride, int horizontal_stride,
      pooling_mode mode,
      bool propagate_nan,
      bool needs_grad)
    : Operation<T>::Operation({input}, needs_grad),
#if defined(MAGMADNN_HAVE_MKLDNN)   
      dnnl_cpu_engine_(dnnl::engine::kind::cpu, 0),
#endif
      input(input),
      filter_h(filter_h), filter_w(filter_w),
      pad_h(pad_h), pad_w(pad_w),
      vertical_stride(vertical_stride), horizontal_stride(horizontal_stride),
      mode(mode),
      propagate_nan(propagate_nan) {

   /* setup code in here */
    this->mem_type = input->get_memory_type();
    this->name = "Pooling";

    /* initialize all the pooling settings */
    this->input_tensor = this->input->get_output_tensor();
    this->init_settings();
}

template <typename T>
PoolingOp<T>::~PoolingOp() {
    if (this->mem_type == HOST) {

// #if defined(MAGMADNN_HAVE_MKLDNN)
//        dnnl_engine_destroy(this->engine_);
// #endif
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {

        cudnnErrchk(cudnnDestroyPoolingDescriptor(this->settings.poolingDesc));
    }
#endif
}

template <typename T>
Tensor<T> *PoolingOp<T>::_eval(bool recompute) {

   input_tensor = input->eval(recompute);

   if (this->mem_type == HOST) {
#if defined(MAGMADNN_HAVE_MKLDNN)

      // Source DNNL memory
      auto src_mem = dnnl::memory(
            this->dnnl_fwd_pdesc_->src_desc(),
            this->dnnl_cpu_engine_,
            // Poiter to underlying source data
            this->input_tensor->get_ptr());

      // Destination DNNL memory
      auto dst_mem = dnnl::memory(
            this->dnnl_fwd_pdesc_->dst_desc(),
            this->dnnl_cpu_engine_,
            // Poiter to underlying destination data
            this->output_tensor->get_ptr());
      
      auto dnnl_workspace_mem = dnnl::memory(
            this->dnnl_fwd_pdesc_->workspace_desc(), this->dnnl_cpu_engine_);

      // Build arg list for kernel execution
      std::unordered_map<int, dnnl::memory> dnnl_args;
      dnnl_args.insert({DNNL_ARG_WORKSPACE, dnnl_workspace_mem});
      dnnl_args.insert({DNNL_ARG_SRC, src_mem});
      dnnl_args.insert({DNNL_ARG_DST, dst_mem});

      // Create dnnl stream.
      dnnl::stream dnnl_engine_stream(this->dnnl_cpu_engine_);
      
      this->dnnl_fwd_->execute(dnnl_engine_stream, dnnl_args);
      dnnl_engine_stream.wait();
#else
      std::fprintf(stderr, "Error: Pooling::_eval requires GPU\n");
#endif
   }
#if defined(MAGMADNN_HAVE_CUDA)
   else {
      this->settings.handle = this->get_cudnn_handle();
      ::magmadnn::math::pooling_device(this->input_tensor, this->output_tensor, this->settings);
      if (!this->get_async()) cudaStreamSynchronize(this->get_custream());
   }
#endif

   return this->output_tensor;
}

template <typename T>
Tensor<T> *PoolingOp<T>::_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad) {
    /* return gradient in here ... */
    Tensor<T> *out = this->_grad_cache[(uintptr_t) var];

#if defined(MAGMADNN_HAVE_MKLDNN)
    dnnl::memory::dims diff_dst_mem_dims;
    dnnl::memory::dims diff_src_mem_dims;
    // mkldnn::memory::dims in_grad_dims;

    dnnl::memory::desc diff_src_mem_md;
    dnnl::memory::desc diff_dst_mem_md;
#endif
    
    if (out == NULL) {
        out = new Tensor<T>(this->input->get_output_shape(), {NONE, {}}, this->mem_type);

#if defined(MAGMADNN_HAVE_CUDA)
        out->set_custream(this->get_custream());
        out->set_cublas_handle(this->get_cublas_handle());
#endif

#if defined(MAGMADNN_HAVE_MKLDNN)
        // Dimensions of destiation gradient 
        diff_dst_mem_dims =
           {
            this->input->get_output_shape()[0],
            this->input->get_output_shape()[1],
            this->input->get_output_shape()[2],
            this->input->get_output_shape()[3]
           };

        diff_dst_mem_md = dnnl::memory::desc(
              diff_dst_mem_dims,
              dnnl::memory::data_type::f32,
              dnnl::memory::format_tag::nchw);

        diff_src_mem_dims =
           {
            grad->get_shape(0),
            grad->get_shape(1),
            grad->get_shape(2),
            grad->get_shape(3)
           };

        diff_src_mem_md = dnnl::memory::desc(
              diff_src_mem_dims,
              dnnl::memory::data_type::f32,
              dnnl::memory::format_tag::nchw);

        dnnl::algorithm pool_alg;

        if (mode == pooling_mode::MAX_POOL) {
           pool_alg = dnnl::algorithm::pooling_max;
        }
        else if (mode == pooling_mode::AVERAGE_POOL) {
           pool_alg = dnnl::algorithm::pooling_avg_exclude_padding;
        }
        else {
           throw ::magmadnn::Error(
                 __FILE__, __LINE__,
                 "Pooling algorithm not supported: " + mode);
        }

        dnnl::pooling_backward::desc desc(
              pool_alg,
              diff_src_mem_md, diff_dst_mem_md,
              {vertical_stride, horizontal_stride},
              {filter_h, filter_w},
              {pad_h, pad_w},
              {pad_h, pad_w});
#endif

        this->_grad_cache[(uintptr_t) var] = out;
    }

    if (this->mem_type == HOST) {
#if defined(MAGMADNN_HAVE_MKLDNN)
       // Source DNNL memory
       auto diff_src_mem = dnnl::memory(
             diff_src_mem_md,
             this->dnnl_cpu_engine_,
             // Poiter to underlying source data
             grad->get_ptr());

       // Destination DNNL memory
       auto diff_dst_mem = dnnl::memory(
             diff_dst_mem_md,
             this->dnnl_cpu_engine_,
             // Poiter to underlying source data
             out->get_ptr());

       // Build arg list for kernel execution
       std::unordered_map<int, dnnl::memory> dnnl_args;
       dnnl_args.insert({DNNL_ARG_SRC, diff_src_mem});
       dnnl_args.insert({DNNL_ARG_DST, diff_dst_mem});        

       // Create dnnl stream.
       dnnl::stream dnnl_engine_stream(this->dnnl_cpu_engine_);
        
       this->dnnl_fwd_->execute(dnnl_engine_stream, dnnl_args);

       dnnl_engine_stream.wait();       
#else
       MAGMADNN_NOT_IMPLEMENTED
#endif
    }

#if defined(MAGMADNN_HAVE_CUDA)
    else {
       this->settings.handle = this->get_cudnn_handle();
       ::magmadnn::math::pooling_grad_device(this->input_tensor, this->output_tensor, grad, out, this->settings);
       if (!this->get_async()) cudaStreamSynchronize(this->get_custream());
    }
#endif

    return out;
}

template <typename T>
void PoolingOp<T>::init_settings() {
    if (this->mem_type == HOST) {
#if defined(MAGMADNN_HAVE_MKLDNN)

       // Compute output dimensions
       int n = 0, c = 0, h = 0, w = 0;

       // Number of batch
       n = this->input_tensor->get_shape(0);
       // Number of chanels
       c = this->input_tensor->get_shape(1);
       // Output height
       h = 1 + (this->input_tensor->get_shape(2) + 2*pad_h - filter_h) / vertical_stride; 
       // Ouput width
       w = 1 + (this->input_tensor->get_shape(3) + 2*pad_w - filter_w) / horizontal_stride;
       
       this->output_shape =
          {
           static_cast<unsigned int>(n),
           static_cast<unsigned int>(c),
           static_cast<unsigned int>(h),
           static_cast<unsigned int>(w)
          };

       // Pooling operation dimensions

       // Input dimension
       dnnl::memory::dims pool_src_dims =
          {
           this->input_tensor->get_shape(0),
           this->input_tensor->get_shape(1),
           this->input_tensor->get_shape(2),
           this->input_tensor->get_shape(3)
          };
       
       // Output dimension
       dnnl::memory::dims pool_dst_dims = 
          {
           this->output_shape[0],
           this->output_shape[1],
           this->output_shape[2],
           this->output_shape[3]
          };

       // Kernel dimension
       dnnl::memory::dims pool_kernel_dims = {filter_h, filter_w};
       // Strides dimension
       dnnl::memory::dims pool_strides_dims = {vertical_stride, horizontal_stride};
       // Padding dimension
       dnnl::memory::dims pool_padding_dims = {pad_h, pad_w};

       // Create memory descriptors
       
       auto pool_src_md = dnnl::memory::desc(
             pool_src_dims,
             dnnl::memory::data_type::f32,
             dnnl::memory::format_tag::nchw);
       // auto pool_src_mem = dnnl::memory(pool_src_md, this->dnnl_cpu_engine_);

       auto pool_dst_md = dnnl::memory::desc(
             pool_dst_dims,
             dnnl::memory::data_type::f32,
             dnnl::memory::format_tag::nchw);
       // auto pool_dst_mem = dnnl::memory(pool_dst_md, this->dnnl_cpu_engine_);

       dnnl::algorithm pool_alg;
       
       if (mode == pooling_mode::MAX_POOL) {
          pool_alg = dnnl::algorithm::pooling_max;
       }
       else if (mode == pooling_mode::AVERAGE_POOL) {
          pool_alg = dnnl::algorithm::pooling_avg_exclude_padding;
       }
       else {
          throw ::magmadnn::Error(
                __FILE__, __LINE__,
                "Pooling algorithm not supported: " + mode);
       }
       
       // Create pooling operation descriptor.
       auto pool_fwd_desc = dnnl::pooling_forward::desc(
             dnnl::prop_kind::forward_training,
             pool_alg,
             pool_src_md, pool_dst_md,
             pool_strides_dims, pool_kernel_dims,
             pool_padding_dims, pool_padding_dims);

       this->dnnl_fwd_pdesc_.reset(
             new dnnl::pooling_forward::primitive_desc(
                   pool_fwd_desc, this->dnnl_cpu_engine_));

       this->dnnl_fwd_.reset(
             new dnnl::pooling_forward(*(this->dnnl_fwd_pdesc_)));

       //
       // Init backward pooling
       
       
       //
       // Create output tensor
       
       // FIXME: call calculate_and_set_output_shape instead? 
       this->output_tensor = new Tensor<T>(this->output_shape, {NONE, {}}, this->mem_type);

#else
       std::fprintf(stderr, "Error: PoolingOp::init_settings requires GPU.\n");
#endif

    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {

       this->settings.handle = this->get_cudnn_handle();
          
       /* init the pooling descriptor */
       cudnnErrchk(cudnnCreatePoolingDescriptor(&this->settings.poolingDesc));

       // std::cout << "filter = " << filter_h << " x " << filter_w << std::endl; 
       // std::cout << "padding = " << pad_h << " x " << pad_w << std::endl; 
       // std::cout << "stride = " << vertical_stride << " x " << horizontal_stride << std::endl; 
       
       /* set the pooling description */
       cudnnErrchk(cudnnSetPooling2dDescriptor(
                         this->settings.poolingDesc,
                         (mode == MAX_POOL) ?
                         /*CUDNN_POOLING_MAX*/ CUDNN_POOLING_MAX_DETERMINISTIC :
                         CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING /*CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING*/,
                         (propagate_nan) ? CUDNN_PROPAGATE_NAN : CUDNN_NOT_PROPAGATE_NAN,
                         filter_h, filter_w, pad_h, pad_w,
                         vertical_stride, horizontal_stride));

       this->calculate_and_set_output_shape();
    }
#endif
}

template <typename T>
void PoolingOp<T>::calculate_and_set_output_shape() {
    /* calculate the correct output shape here */
    if (this->mem_type == HOST) {
        std::fprintf(stderr, "Error: Pooling::output_shape requires GPU.\n");
        this->output_shape = this->input->get_output_shape();
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {
        int n = 0, c = 0, h = 0, w = 0;

        cudnnTensorDescriptor_t cudnn_tensor_desc = this->input_tensor->get_cudnn_tensor_descriptor();

        cudnnDataType_t data_type;

        // int nStride = 0, cStride = 0, hStride = 0, wStride = 0;
           
        // cudnnGetTensor4dDescriptor(
        //       cudnn_tensor_desc,
        //       &data_type,
        //       &n, &c, &h, &w,
        //       &nStride, &cStride, &hStride, &wStride);

        // std::cout << "Input tensor:" << std::endl; 
        // std::cout << "n = " << n << std::endl; 
        // std::cout << "c = " << c << std::endl; 
        // std::cout << "h = " << h << std::endl; 
        // std::cout << "w = " << w << std::endl; 

        // std::cout << "nStride = " << nStride << std::endl; 
        // std::cout << "cStride = " << cStride << std::endl; 
        // std::cout << "hStride = " << hStride << std::endl; 
        // std::cout << "wStride = " << wStride << std::endl; 

        
        cudnnErrchk(
              cudnnGetPooling2dForwardOutputDim(
                    this->settings.poolingDesc,
                    cudnn_tensor_desc,
                    &n, &c, &h, &w));

        assert((n >= 0) && (c >=0) && (h >= 0) && (w >= 0));

        // std::cout << "Pooling forward outdim:" << std::endl; 
        // std::cout << "n = " << n << std::endl; 
        // std::cout << "c = " << c << std::endl; 
        // std::cout << "h = " << h << std::endl; 
        // std::cout << "w = " << w << std::endl; 

        this->output_shape = {static_cast<unsigned int>(n),
                              static_cast<unsigned int>(c),
                              static_cast<unsigned int>(h),
                              static_cast<unsigned int>(w)};
    }
#endif

    this->output_tensor = new Tensor<T>(this->output_shape, {NONE, {}}, this->mem_type);
#if defined(MAGMADNN_HAVE_CUDA)
    this->output_tensor->set_custream(this->get_custream());
    this->output_tensor->set_cublas_handle(this->get_cublas_handle());
#endif
    
}

template class PoolingOp<int>;
template class PoolingOp<float>;
template class PoolingOp<double>;

template <typename T>
PoolingOp<T> *pooling(Operation<T> *input, int filter_h, int filter_w, int pad_h, int pad_w, int vertical_stride,
                      int horizontal_stride, pooling_mode mode, bool propagate_nan, bool needs_grad) {
    return new PoolingOp<T>(input, filter_h, filter_w, pad_h, pad_w, vertical_stride, horizontal_stride, mode,
                            propagate_nan, needs_grad);
}
template PoolingOp<int> *pooling(Operation<int> *input, int filter_h, int filter_w, int pad_h, int pad_w,
                                 int vertical_stride, int horizontal_stride, pooling_mode mode, bool propagate_nan,
                                 bool needs_grad);
template PoolingOp<float> *pooling(Operation<float> *input, int filter_h, int filter_w, int pad_h, int pad_w,
                                   int vertical_stride, int horizontal_stride, pooling_mode mode, bool propagate_nan,
                                   bool needs_grad);
template PoolingOp<double> *pooling(Operation<double> *input, int filter_h, int filter_w, int pad_h, int pad_w,
                                    int vertical_stride, int horizontal_stride, pooling_mode mode, bool propagate_nan,
                                    bool needs_grad);

}  // namespace op
}  // namespace magmadnn
