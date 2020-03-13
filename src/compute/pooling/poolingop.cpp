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
      input(input),
      filter_h(filter_h),
      filter_w(filter_w),
      pad_h(pad_h),
      pad_w(pad_w),
      vertical_stride(vertical_stride),
      horizontal_stride(horizontal_stride),
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

       dnnl_memory_desc_t pool_src_md;

       dnnl_dim_t pool_src_sizes[4] =
          {
           this->input_tensor->get_shape(0),
           this->input_tensor->get_shape(1),
           this->input_tensor->get_shape(2),
           this->input_tensor->get_shape(3)
          };

       // Create memory descriptor for source
       dnnl_memory_desc_init_by_tag(
             &pool_src_md,
             4, pool_src_sizes, // Output dimensions
             dnnl_f32, // Datatype
             dnnl_format_tag_t::dnnl_format_tag_any);

       // dnnl_memory_desc_t pool_dst_md;
       // dnnl_dim_t pool_dst_sizes[4] = {}:

       // Create memory descriptor for destination
       // dnnl_memory_desc_init_by_tag(
       //       &pool_dst_md,
       //       4, pool_dst_sizes, // Output dimensions
       //       dnnl_f32, // Datatype
       //       dnnl_format_tag_t::dnnl_format_tag_any);
       
       // dnnl_primitive_desc_t pool_pd;
       // dnnl_pooling_desc_t pool_desc;

       dnnl_dim_t pool_strides[2] = {vertical_stride, horizontal_stride};
          
       // dnnl_pooling_forward_desc_init(
       //       &pool_desc,  dnnl_prop_kind_t::dnnl_forward,
       //       dnnl_alg_kind_t::dnnl_pooling_max, pool_src_md, &pool_dst_md, pool_strides,
       //       pool_kernel, pool_padding, pool_padding));
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
