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
      std::fprintf(stderr, "Error: Pooling::_eval requires GPU\n");
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

    if (out == NULL) {
        out = new Tensor<T>(this->input->get_output_shape(), {NONE, {}}, this->mem_type);
#if defined(MAGMADNN_HAVE_CUDA)
        out->set_custream(this->get_custream());
        out->set_cublas_handle(this->get_cublas_handle());
#endif
        this->_grad_cache[(uintptr_t) var] = out;
    }

    if (this->mem_type == HOST) {
        ::magmadnn::math::pooling_grad(this->input_tensor, this->output_tensor, grad, out);
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
       
       this->output_shape = {static_cast<unsigned int>(n),
                             static_cast<unsigned int>(c),
                             static_cast<unsigned int>(h),
                             static_cast<unsigned int>(w)};

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

       // kernel dimension
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
       auto pool_src_mem = dnnl::memory(pool_src_md, this->dnnl_cpu_engine_);

       auto pool_dst_md = dnnl::memory::desc(
             pool_dst_dims,
             dnnl::memory::data_type::f32,
             dnnl::memory::format_tag::nchw);
       auto pool_dst_mem = dnnl::memory(pool_dst_md, this->dnnl_cpu_engine_);

       
       
       // create a pooling primitive descriptor
       // auto pool_fwd_desc = pooling_forward::desc(
       //       prop_kind::forward,
       //       algorithm::pooling_max, lrn_dst_memory.get_desc(), pool_dst_md,
       //       pool_strides, pool_kernel, pool_padding, pool_padding);
       // auto pool_fwd_pd = pooling_forward::primitive_desc(
       //       pool_fwd_desc, this->dnnl_cpu_engine_);
       
       // // Crate DNNL engine
       // dnnl_engine_create(
       //       &this->engine_,
       //       dnnl_cpu, // Engine type (CPU, GPU or unspecified)
       //       0 // Engine index
       //       );
       
       // dnnl_status_t dnnl_stat = dnnl_success;
          
       // dnnl_memory_desc_t pool_src_md;

       // dnnl_dim_t pool_src_sizes[4] =
       //    {
       //     this->input_tensor->get_shape(0),
       //     this->input_tensor->get_shape(1),
       //     this->input_tensor->get_shape(2),
       //     this->input_tensor->get_shape(3)
       //    };

       // // Create memory descriptor for source
       // dnnl_stat = dnnl_memory_desc_init_by_tag(
       //       &pool_src_md,
       //       4, pool_src_sizes, // Output dimensions
       //       // TODO: adapt dnnl datatype to T
       //       dnnl_f32, // Datatype
       //       dnnl_format_tag_t::dnnl_format_tag_any);
       // MAGMADNN_ASSERT_NO_DNNL_ERRORS(dnnl_stat);
          
       // dnnl_memory_desc_t pool_dst_md;

       // dnnl_dim_t pool_dst_sizes[4] =
       //    {
       //     this->output_shape[0],
       //     this->output_shape[1],
       //     this->output_shape[2],
       //     this->output_shape[3]
       //    };

       // // Create memory descriptor for destination
       // dnnl_stat = dnnl_memory_desc_init_by_tag(
       //       &pool_dst_md,
       //       4, pool_dst_sizes, // Output dimensions
       //       dnnl_f32, // Datatype
       //       dnnl_format_tag_t::dnnl_format_tag_any);
       // MAGMADNN_ASSERT_NO_DNNL_ERRORS(dnnl_stat);
       
       // dnnl_dim_t pool_strides[2] = {vertical_stride, horizontal_stride};

       // dnnl_alg_kind_t alg_kind = dnnl_alg_kind_t::dnnl_pooling_max;

       // if (mode == MAX_POOL) {
       //    alg_kind = dnnl_alg_kind_t::dnnl_pooling_max;
       // }
       // else if (mode == AVERAGE_POOL) {
       //    alg_kind = dnnl_alg_kind_t::dnnl_pooling_avg_exclude_padding;
       // }
       // else {
       //    throw ::magmadnn::Error(
       //          __FILE__, __LINE__,
       //          "Pooling algorithm not supported: " + mode);
       // }

       // // Filter dimensions
       // dnnl_dim_t pool_kernel[2] = {filter_h, filter_w};
       // // Padding dimensions
       // dnnl_dim_t pool_padding[2] = {pad_h, pad_w};

       // //
       // // Init pooling forward

       // // Create pooling operation descriptor
       // dnnl_stat = dnnl_pooling_forward_desc_init(
       //       &this->dnnl_pool_fwd_desc_,  dnnl_prop_kind_t::dnnl_forward,
       //       alg_kind,
       //       &pool_src_md, &pool_dst_md,
       //       pool_strides,
       //       pool_kernel,
       //       pool_padding, pool_padding);
       // MAGMADNN_ASSERT_NO_DNNL_ERRORS(dnnl_stat);

       // // Create primitive descriptor
       // dnnl_primitive_desc_t pool_fwd_pd;

       // dnnl_stat = dnnl_primitive_desc_create(
       //       &pool_fwd_pd, &this->dnnl_pool_fwd_desc_, NULL, this->engine_, NULL);
       // MAGMADNN_ASSERT_NO_DNNL_ERRORS(dnnl_stat);

       // // create memory for workspace
       // dnnl_memory_t pool_ws_memory;
       
       // const dnnl_memory_desc_t *pool_ws_md = dnnl_primitive_desc_query_md(
       //       pool_fwd_pd, dnnl_query_workspace_md, 0);

       // dnnl_stat = dnnl_memory_create(
       //       &pool_ws_memory, pool_ws_md, this->engine_, DNNL_MEMORY_ALLOCATE);
       // MAGMADNN_ASSERT_NO_DNNL_ERRORS(dnnl_stat);
       
       // //
       // // Init pooling backward
       
       // dnnl_memory_desc_t const pool_diff_src_md = pool_src_md;
       // // pooling diff dst memory descriptor
       // dnnl_memory_desc_t const pool_diff_dst_md = pool_dst_md;       

       // dnnl_stat = dnnl_pooling_backward_desc_init(
       //       &this->dnnl_pool_bwd_desc_,
       //       alg_kind,
       //       &pool_diff_src_md, &pool_diff_dst_md,
       //       pool_strides,
       //       pool_kernel,
       //       pool_padding, pool_padding);
       // MAGMADNN_ASSERT_NO_DNNL_ERRORS(dnnl_stat);
       
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
