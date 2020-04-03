#include "compute/conv2dforward/conv2dforwardop.h"

#include <iostream>

#include "magmadnn/config.h"

namespace magmadnn {
namespace op {

template <typename T>
Conv2DForwardOp<T>::Conv2DForwardOp(
      Operation<T> *input, Operation<T> *filter, int pad_h, int pad_w,
      int vertical_stride, int horizontal_stride, int dilation_h, int dilation_w,
      bool use_cross_correlation, bool needs_grad)
    : Operation<T>::Operation({input, filter}, needs_grad),
      input(input),
      filter(filter),
#if defined(MAGMADNN_HAVE_MKLDNN)   
      dnnl_cpu_engine_(dnnl::engine::kind::cpu, 0),
      dnnl_fwd_pdesc_(nullptr),
#endif
      pad_h(pad_h),
      pad_w(pad_w),
      vertical_stride(vertical_stride),
      horizontal_stride(horizontal_stride),
      dilation_h(dilation_h),
      dilation_w(dilation_w),
      use_cross_correlation(use_cross_correlation) {
   
    /* setup code in here */
    this->mem_type = input->get_memory_type();

    /* initialize all the conv settings */
    this->input_tensor = this->input->get_output_tensor();
    this->init_settings();

#if defined(MAGMADNN_HAVE_MKLDNN)   
    this->init_dnnl_settings();
#endif
}

template <typename T>
Conv2DForwardOp<T>::~Conv2DForwardOp() {
    if (this->mem_type == HOST) {
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {

        cudaErrchk(cudaFree(this->cudnn_settings.workspace));
        cudaErrchk(cudaFree(this->cudnn_settings.grad_data_workspace));
        cudaErrchk(cudaFree(this->cudnn_settings.grad_filter_workspace));

        cudnnErrchk(cudnnDestroyFilterDescriptor(this->cudnn_settings.filter_desc));
        cudnnErrchk(cudnnDestroyConvolutionDescriptor(this->cudnn_settings.conv_desc));
    }
#endif
}

template <typename T>
Tensor<T> *Conv2DForwardOp<T>::_eval(bool recompute) {

   this->input_tensor = input->eval(recompute);
   this->filter_tensor = filter->eval(recompute);

   if (this->mem_type == HOST) {
#if defined(MAGMADNN_HAVE_MKLDNN)

      // dnnl::engine eng(dnnl::engine::kind::cpu, 0);

      // dnnl::memory::dims src_dims =
      //    {this->input_tensor->get_shape(0), this->input_tensor->get_shape(1),
      //     this->input_tensor->get_shape(2), this->input_tensor->get_shape(3)};

      // dnnl::memory::dims dst_dims =
      //    {this->output_tensor->get_shape(0), this->output_tensor->get_shape(1),
      //     this->output_tensor->get_shape(2), this->output_tensor->get_shape(3)};

      // dnnl::memory::dims weights_dims =
      //    {this->filter_tensor->get_shape(0), this->filter_tensor->get_shape(1),
      //     this->filter_tensor->get_shape(2), this->filter_tensor->get_shape(3)};

      // auto src_md = dnnl::memory::desc(
      //       src_dims,
      //       dnnl::memory::data_type::f32,
      //       dnnl::memory::format_tag::nchw);

      // auto dst_md = dnnl::memory::desc(
      //       dst_dims,
      //       dnnl::memory::data_type::f32,
      //       dnnl::memory::format_tag::nchw);

      // auto weights_md = dnnl::memory::desc(
      //       weights_dims,
      //       dnnl::memory::data_type::f32,
      //       dnnl::memory::format_tag::nchw);

      // auto bias_md = dnnl::memory::desc();
            
      auto src_md = this->dnnl_fwd_pdesc_->src_desc();
      auto dst_md = this->dnnl_fwd_pdesc_->dst_desc();
      auto bias_md = this->dnnl_fwd_pdesc_->bias_desc();
      auto weights_md = this->dnnl_fwd_pdesc_->weights_desc();

      // // Strides dimension
      // dnnl::memory::dims conv_strides_dims = {vertical_stride, horizontal_stride};
      // // Padding dimension
      // dnnl::memory::dims conv_padding_dims = {pad_h, pad_w};
      // // Dilatation dimension
      // dnnl::memory::dims conv_dilation_dims = {dilation_h, dilation_w};

      // dnnl::algorithm conv_alg;

      // auto conv_fwd_desc = dnnl::convolution_forward::desc(
      //       dnnl::prop_kind::forward_training,
      //       dnnl::algorithm::convolution_direct,
      //       src_md, weights_md, bias_md, dst_md,
      //       conv_strides_dims, conv_dilation_dims,
      //       conv_padding_dims, conv_padding_dims);

      // auto conv_fwd_pdesc = dnnl::convolution_forward::primitive_desc(
      //       conv_fwd_desc, eng);

      // auto conv_fwd = dnnl::convolution_forward(
      //       conv_fwd_pdesc);

      auto src_mem = dnnl::memory(
            src_md,
            // eng,
            this->dnnl_cpu_engine_,
            this->input_tensor->get_ptr());
      auto dst_mem = dnnl::memory(
            dst_md,
            this->dnnl_cpu_engine_,
            // eng,
            this->output_tensor->get_ptr());
      auto bias_mem = dnnl::memory(
            bias_md,
            // eng,
            this->dnnl_cpu_engine_,
            nullptr);
      auto weights_mem = dnnl::memory(
            weights_md,
            this->dnnl_cpu_engine_,
            // eng,
            this->filter_tensor->get_ptr());

      // Primitive arguments.
      std::unordered_map<int, dnnl::memory> conv_fwd_args;
      conv_fwd_args.insert({DNNL_ARG_SRC, src_mem});
      conv_fwd_args.insert({DNNL_ARG_WEIGHTS, weights_mem});
      conv_fwd_args.insert({DNNL_ARG_BIAS, bias_mem});
      conv_fwd_args.insert({DNNL_ARG_DST, dst_mem});

      // std::cout << "eval" << std::endl;

      // Create dnnl::stream.
      dnnl::stream engine_stream(this->dnnl_cpu_engine_);
      // dnnl::stream engine_stream(eng);
      dnnl_fwd_->execute(engine_stream, conv_fwd_args);
      // conv_fwd.execute(engine_stream, conv_fwd_args);
      // Wait for the computation to finalize.
      engine_stream.wait();

      // std::cout << "eval end" << std::endl;
#else          
      std::fprintf(stderr, "Error: Conv2dForward::_eval requires GPU\n");
#endif
   }
#if defined(MAGMADNN_HAVE_CUDA)
   else {
      this->cudnn_settings.handle = this->get_cudnn_handle();
      ::magmadnn::math::conv2d_device(
            this->input_tensor, this->filter_tensor, this->output_tensor,
            this->cudnn_settings);

      if (!this->get_async()) cudaStreamSynchronize(this->get_custream());
   }
#endif

   return this->output_tensor;
}

template <typename T>
Tensor<T> *Conv2DForwardOp<T>::_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad) {

   /* return gradient in here ... */
   Tensor<T> *out = this->_grad_cache[(uintptr_t) var];

   if (var == this->input) {
      if (out == NULL) {
         out = new Tensor<T>(this->input->get_output_shape(), {NONE, {}}, this->mem_type);
#if defined(MAGMADNN_HAVE_CUDA)
         out->set_custream(this->get_custream());
         out->set_cublas_handle(this->get_cublas_handle());
#endif
         this->_grad_cache[(uintptr_t) var] = out;
      }

      this->filter_tensor = this->filter->eval(false);

      if (this->mem_type == HOST) {
#if defined(MAGMADNN_HAVE_MKLDNN)

         dnnl::memory::dims diff_src_dims =
            {out->get_shape(0), out->get_shape(1),
             out->get_shape(2), out->get_shape(3)};

         dnnl::memory::dims diff_dst_dims =
            {grad->get_shape(0), grad->get_shape(1),
             grad->get_shape(2), grad->get_shape(3)};

         dnnl::memory::desc diff_src_md = dnnl::memory::desc(
               diff_src_dims,
               dnnl::memory::data_type::f32,
               dnnl::memory::format_tag::nchw);

         dnnl::memory::desc diff_dst_md = dnnl::memory::desc(
               diff_dst_dims,
               dnnl::memory::data_type::f32,
               dnnl::memory::format_tag::nchw);

         auto weights_md = this->dnnl_fwd_pdesc_->weights_desc();
         
         // Strides dimension
         dnnl::memory::dims conv_strides_dims = {vertical_stride, horizontal_stride};
         // Padding dimension
         dnnl::memory::dims conv_padding_dims = {pad_h, pad_w};
         // Dilatation dimension
         dnnl::memory::dims conv_dilation_dims = {dilation_h, dilation_w};

         auto conv_bwd_data_desc = dnnl::convolution_backward_data::desc(
               dnnl::algorithm::convolution_direct, diff_src_md, weights_md,
               diff_src_md, conv_strides_dims, conv_dilation_dims,
               conv_padding_dims, conv_padding_dims);

         auto conv_bwd_data_pdesc =
            dnnl::convolution_backward_data::primitive_desc(
               conv_bwd_data_desc, this->dnnl_cpu_engine_,
               *(this->dnnl_fwd_pdesc_.get()));

         auto conv_bwd_data =
            dnnl::convolution_backward_data(conv_bwd_data_pdesc); 

         // auto bias_mem = dnnl::memory(bias_md, this->dnnl_cpu_engine_, nullptr);
         auto weights_mem = dnnl::memory(
               weights_md, this->dnnl_cpu_engine_, this->filter_tensor->get_ptr());

         auto diff_src_mem = dnnl::memory(
               diff_src_md, this->dnnl_cpu_engine_, out->get_ptr());

         auto diff_dst_mem = dnnl::memory(
               diff_dst_md, this->dnnl_cpu_engine_, grad->get_ptr());

         // Primitive arguments.
         std::unordered_map<int, dnnl::memory> conv_bwd_data_args;
         conv_bwd_data_args.insert({DNNL_ARG_DIFF_SRC, diff_src_mem});
         conv_bwd_data_args.insert({DNNL_ARG_WEIGHTS, weights_mem});
         conv_bwd_data_args.insert({DNNL_ARG_DIFF_DST, diff_dst_mem});

         // Create dnnl::stream.
         dnnl::stream engine_stream(this->dnnl_cpu_engine_);
         conv_bwd_data.execute(engine_stream, conv_bwd_data_args);
         // Wait for the computation to finalize.
         engine_stream.wait();

#else
         ::magmadnn::math::conv2d_grad_data(this->filter_tensor, grad, out);
#endif
      }
#if defined(MAGMADNN_HAVE_CUDA)
      else {
         this->cudnn_settings.handle = this->get_cudnn_handle();
         ::magmadnn::math::conv2d_grad_data_device(
               this->filter_tensor, grad, out, this->cudnn_settings);
         if (!this->get_async()) cudaStreamSynchronize(this->get_custream());

      }
#endif

   }
   else if (var == this->filter) {

      if (out == NULL) {
         out = new Tensor<T>(this->filter->get_output_shape(), {NONE, {}}, this->mem_type);
#if defined(MAGMADNN_HAVE_CUDA)
         out->set_custream(this->get_custream());
         out->set_cublas_handle(this->get_cublas_handle());
#endif
         this->_grad_cache[(uintptr_t) var] = out;
      }

      this->input_tensor = this->input->eval(false);

      if (this->mem_type == HOST) {
#if defined(MAGMADNN_HAVE_MKLDNN)
         dnnl::memory::dims diff_weigths_dims =
            {out->get_shape(0), out->get_shape(1),
             out->get_shape(2), out->get_shape(3)};

         dnnl::memory::dims diff_dst_dims =
            {grad->get_shape(0), grad->get_shape(1),
             grad->get_shape(2), grad->get_shape(3)};

         dnnl::memory::desc diff_weights_md = dnnl::memory::desc(
               diff_weigths_dims,
               dnnl::memory::data_type::f32,
               dnnl::memory::format_tag::nchw);

         dnnl::memory::desc diff_dst_md = dnnl::memory::desc(
               diff_dst_dims,
               dnnl::memory::data_type::f32,
               dnnl::memory::format_tag::nchw);

         // dnnl::memory::desc diff_bias_md = dnnl::memory::desc(
         //       {0,0},
         //       dnnl::memory::data_type::f32,
         //       dnnl::memory::format_tag::nchw);

         // Create a zero memory descriptor
         dnnl::memory::desc diff_bias_md = dnnl::memory::desc();
         
         auto src_md = this->dnnl_fwd_pdesc_->src_desc();

         // Strides dimension
         dnnl::memory::dims conv_strides_dims = {vertical_stride, horizontal_stride};
         // Padding dimension
         dnnl::memory::dims conv_padding_dims = {pad_h, pad_w};
         // Dilatation dimension
         dnnl::memory::dims conv_dilation_dims = {dilation_h, dilation_w};

         auto conv_bwd_weights_desc = dnnl::convolution_backward_weights::desc(
               dnnl::algorithm::convolution_direct, src_md, diff_weights_md,
               diff_bias_md, diff_dst_md, conv_strides_dims, conv_dilation_dims,
               conv_padding_dims, conv_padding_dims);

         auto conv_bwd_weights_pdesc =
            dnnl::convolution_backward_weights::primitive_desc(
                  conv_bwd_weights_desc, this->dnnl_cpu_engine_,
                  *(this->dnnl_fwd_pdesc_.get()));

         auto conv_bwd_weights =
            dnnl::convolution_backward_weights(conv_bwd_weights_pdesc); 

         // auto bias_mem = dnnl::memory(bias_md, this->dnnl_cpu_engine_, nullptr);
         auto diff_weights_mem = dnnl::memory(
               diff_weights_md, this->dnnl_cpu_engine_,
               out->get_ptr());

         auto src_mem = dnnl::memory(
               src_md, this->dnnl_cpu_engine_, this->input_tensor->get_ptr());

         auto diff_dst_mem = dnnl::memory(
               diff_dst_md, this->dnnl_cpu_engine_, grad->get_ptr());

         auto diff_bias_mem = dnnl::memory(
               diff_bias_md, this->dnnl_cpu_engine_, nullptr);

         // Primitive arguments.
         std::unordered_map<int, dnnl::memory> conv_bwd_weights_args;
         conv_bwd_weights_args.insert({DNNL_ARG_SRC, src_mem});
         conv_bwd_weights_args.insert({DNNL_ARG_DIFF_WEIGHTS, diff_weights_mem});
         conv_bwd_weights_args.insert({DNNL_ARG_DIFF_BIAS, diff_bias_mem});
         conv_bwd_weights_args.insert({DNNL_ARG_DIFF_DST, diff_dst_mem});

         // Create dnnl::stream.
         dnnl::stream engine_stream(this->dnnl_cpu_engine_);
         conv_bwd_weights.execute(engine_stream, conv_bwd_weights_args);
         // Wait for the computation to finalize.
         engine_stream.wait();

#else
         ::magmadnn::math::conv2d_grad_filter(this->input_tensor, grad, out);
#endif
      }
#if defined(MAGMADNN_HAVE_CUDA)
      else {
         this->cudnn_settings.handle = this->get_cudnn_handle();
         ::magmadnn::math::conv2d_grad_filter_device(
               this->input_tensor, grad, out, this->cudnn_settings);
         if (!this->get_async()) cudaStreamSynchronize(this->get_custream());
      }
#endif

   } else {
      std::fprintf(stderr, "Error: bad conv2d grad\n");
   }

   return out;
}

#if defined(MAGMADNN_HAVE_MKLDNN)
template <typename T>
void Conv2DForwardOp<T>::init_dnnl_settings() {

   // std::cout << "Conv2DForwardOp<T>::init_dnnl_settings, get_count = "
   //           << dnnl::engine::get_count(dnnl::engine::kind::cpu)
   //           << std::endl;
   
   int in = 0, ic = 0, ih = 0, iw = 0;

   in = this->input_tensor->get_shape(0);
   ic = this->input_tensor->get_shape(1);
   ih = this->input_tensor->get_shape(2);
   iw = this->input_tensor->get_shape(3);
      
   dnnl::memory::dims src_dims = {in, ic, ih, iw};

   int kh = this->filter->get_output_shape()[2];
   int kw = this->filter->get_output_shape()[3];
   
   // Calculate convolution output shape
   int on = 0, oc = 0, oh = 0, ow = 0;

   on = in;
   oc = this->filter->get_output_shape()[0];

   int dkh = 1 + (kh-1)*(dilation_h + 1); 
   int dkw = 1 + (kw-1)*(dilation_w + 1); 
      
   oh = 1 + (ih - dkh + 2*pad_h) / vertical_stride; 
   ow = 1 + (iw - dkw + 2*pad_w) / horizontal_stride; 
   
   dnnl::memory::dims dst_dims = {on, oc, oh, ow};

   // FIXME: Set output_shape outside this routine?
   this->calculate_and_set_output_shape();

   dnnl::memory::dims weights_dims = {oc, ic, kh, kw};

   auto src_md = dnnl::memory::desc(
         src_dims,
         dnnl::memory::data_type::f32,
         dnnl::memory::format_tag::nchw);

   auto dst_md = dnnl::memory::desc(
         dst_dims,
         dnnl::memory::data_type::f32,
         dnnl::memory::format_tag::nchw);
   
   auto weights_md = dnnl::memory::desc(
         weights_dims,
         dnnl::memory::data_type::f32,
         dnnl::memory::format_tag::nchw);

   // TODO: Add bias
   // auto bias_md = dnnl::memory::desc(
   //       {0,0,0,0},
   //       dnnl::memory::data_type::f32,
   //       dnnl::memory::format_tag::nchw);
   // Create a zero memory descriptor
   auto bias_md = dnnl::memory::desc();

   // Strides dimension
   dnnl::memory::dims conv_strides_dims = {vertical_stride, horizontal_stride};
   // Padding dimension
   dnnl::memory::dims conv_padding_dims = {pad_h, pad_w};
   // Dilatation dimension
   dnnl::memory::dims conv_dilation_dims = {dilation_h, dilation_w};

   // dnnl::algorithm conv_alg;

   dnnl::convolution_forward::desc conv_desc =
      dnnl::convolution_forward::desc(
            dnnl::prop_kind::forward_training,
            dnnl::algorithm::convolution_direct,
            src_md, weights_md, bias_md, dst_md,
            conv_strides_dims, conv_dilation_dims,
            conv_padding_dims, conv_padding_dims);

   // dnnl::convolution_forward::desc conv_desc =
   //    dnnl::convolution_forward::desc(
   //          dnnl::prop_kind::forward_training,
   //          dnnl::algorithm::convolution_direct,
   //          src_md, weights_md, dst_md,
   //          conv_strides_dims,
   //          conv_padding_dims, conv_padding_dims);

   this->dnnl_fwd_pdesc_.reset(
         new dnnl::convolution_forward::primitive_desc(
               conv_desc, this->dnnl_cpu_engine_));

   this->dnnl_fwd_.reset(
         new dnnl::convolution_forward(
               *(this->dnnl_fwd_pdesc_.get())));
}
#endif
   
template <typename T>
void Conv2DForwardOp<T>::init_settings() {
    if (this->mem_type == HOST) {
#if !defined(MAGMADNN_HAVE_MKLDNN)
       std::fprintf(stderr, "Error: Conv2DForward::init_settings requires GPU.\n");
#endif
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {

        this->cudnn_settings.handle = this->get_cudnn_handle();
       
        /* init the conv descriptor */
        cudnnErrchk(cudnnCreateConvolutionDescriptor(&this->cudnn_settings.conv_desc));

        // std::cout << "Padding: " << pad_h << "x" << pad_w << std::endl;
        // std::cout << "Stride: " << vertical_stride << "x" << horizontal_stride << std::endl;

        assert((vertical_stride > 0) && (horizontal_stride > 0));
        
        /* set the convolution description */
        cudnnErrchk(cudnnSetConvolution2dDescriptor(
            this->cudnn_settings.conv_desc, pad_h, pad_w, vertical_stride, horizontal_stride, dilation_h, dilation_w,
            (use_cross_correlation) ? CUDNN_CROSS_CORRELATION : CUDNN_CONVOLUTION,
            ::magmadnn::internal::get_cudnn_data_type(static_cast<T>(0))));

        /* init and create the filter descriptor */
        int filter_dims[4];
        const std::vector<unsigned int> &filter_shape = this->filter->get_output_shape();
        for (unsigned int i = 0; i < 4; i++) {
            if (i >= filter_shape.size())
                filter_dims[i] = 1;
            else
                filter_dims[i] = filter_shape[i];
        }
        cudnnErrchk(cudnnCreateFilterDescriptor(&this->cudnn_settings.filter_desc));
        cudnnErrchk(cudnnSetFilter4dDescriptor(
            this->cudnn_settings.filter_desc, ::magmadnn::internal::get_cudnn_data_type(static_cast<T>(0)),
            CUDNN_TENSOR_NCHW, filter_dims[0], filter_dims[1], filter_dims[2], filter_dims[3]));

        this->calculate_and_set_output_shape();

        /* use CUDNN to get the correct/optimal convolution algorithm */
        cudnnErrchk(cudnnGetConvolutionForwardAlgorithm(
            this->cudnn_settings.handle,
            this->input->get_output_tensor()->get_cudnn_tensor_descriptor(), this->cudnn_settings.filter_desc,
            this->cudnn_settings.conv_desc, this->output_tensor->get_cudnn_tensor_descriptor(),
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &this->cudnn_settings.algo));

        /* use CuDNN to get the necessary workspace size and allocate that memory */
        cudnnErrchk(cudnnGetConvolutionForwardWorkspaceSize(
            this->cudnn_settings.handle,
            this->input->get_output_tensor()->get_cudnn_tensor_descriptor(), this->cudnn_settings.filter_desc,
            this->cudnn_settings.conv_desc, this->output_tensor->get_cudnn_tensor_descriptor(),
            this->cudnn_settings.algo, &this->cudnn_settings.workspace_size));
        // std::cout << "Conv2DForwardOp<T>::init_settings, forward workspace size (MB) = "
        //           << (float) ((float) this->cudnn_settings.workspace_size / ((float) 1024.0 * 1024.0))  << std::endl;
        cudaErrchk(cudaMalloc((void **) &this->cudnn_settings.workspace, this->cudnn_settings.workspace_size));

        /* INIT the grad settings */
        cudnnErrchk(cudnnGetConvolutionBackwardDataAlgorithm(
            this->cudnn_settings.handle,
            this->cudnn_settings.filter_desc,
            this->output_tensor->get_cudnn_tensor_descriptor(),                                /* use output for dy */
            this->cudnn_settings.conv_desc, this->input_tensor->get_cudnn_tensor_descriptor(), /* use input for dx */
            CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &this->cudnn_settings.bwd_data_algo));

        cudnnErrchk(cudnnGetConvolutionBackwardFilterAlgorithm(
            this->cudnn_settings.handle,
            this->input_tensor->get_cudnn_tensor_descriptor(),
            this->output_tensor->get_cudnn_tensor_descriptor(), this->cudnn_settings.conv_desc,
            this->cudnn_settings.filter_desc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0,
            &this->cudnn_settings.bwd_filter_algo));

        /* get the workspaces for each of the backward algorithms */
        cudnnErrchk(cudnnGetConvolutionBackwardDataWorkspaceSize(
            this->cudnn_settings.handle,
            this->cudnn_settings.filter_desc,
            this->output_tensor->get_cudnn_tensor_descriptor(), this->cudnn_settings.conv_desc,
            this->input_tensor->get_cudnn_tensor_descriptor(), this->cudnn_settings.bwd_data_algo,
            &this->cudnn_settings.grad_data_workspace_size));
        // std::cout << "Conv2DForwardOp<T>::init_settings, backward workspace size (MB) = "
        //           << (float) ((float) this->cudnn_settings.grad_data_workspace_size / ((float) 1024.0 * 1024.0))
        //           << std::endl;
        cudaErrchk(cudaMalloc((void **) &this->cudnn_settings.grad_data_workspace,
                              this->cudnn_settings.grad_data_workspace_size));

        cudnnErrchk(cudnnGetConvolutionBackwardFilterWorkspaceSize(
            this->cudnn_settings.handle, this->input_tensor->get_cudnn_tensor_descriptor(),
            this->output_tensor->get_cudnn_tensor_descriptor(), this->cudnn_settings.conv_desc,
            this->cudnn_settings.filter_desc, this->cudnn_settings.bwd_filter_algo,
            &this->cudnn_settings.grad_filter_workspace_size));
        // std::cout << "Conv2DForwardOp<T>::init_settings, filter workspace size (MB) = "
        //           << (float) ((float) this->cudnn_settings.grad_filter_workspace_size / ((float) 1024.0 * 1024.0))
        //           << std::endl;
        cudaErrchk(cudaMalloc((void **) &this->cudnn_settings.grad_filter_workspace,
                              this->cudnn_settings.grad_filter_workspace_size));

        // std::cout << "Conv2DForwardOp<T>::init_settings, cuDNN workspace size (MB) = "
        //           << (float) ((float) (this->cudnn_settings.grad_filter_workspace_size
        //                                + this->cudnn_settings.grad_data_workspace_size
        //                                + this->cudnn_settings.workspace_size) / ((float) 1024.0 * 1024.0))
        //           << std::endl;

    }
#endif
}

template <typename T>
void Conv2DForwardOp<T>::calculate_and_set_output_shape() {

   int on = 0, oc = 0, oh = 0, ow = 0;

   /* calculate the correct output shape here */
    if (this->mem_type == HOST) {
#if defined(MAGMADNN_HAVE_MKLDNN)

       int in = 0, ic = 0, ih = 0, iw = 0;

       in = this->input_tensor->get_shape(0);
       ic = this->input_tensor->get_shape(1);
       ih = this->input_tensor->get_shape(2);
       iw = this->input_tensor->get_shape(3);

       // Filter kernel dimensions 
       int kh = this->filter->get_output_shape()[2];
       int kw = this->filter->get_output_shape()[3];

       int dkh = 1 + (kh-1)*(dilation_h + 1); 
       int dkw = 1 + (kw-1)*(dilation_w + 1); 

       // Calculate convolution output shape       
       on = in;
       oc = this->filter->get_output_shape()[0];
       oh = 1 + (ih - dkh + 2*pad_h) / vertical_stride; 
       ow = 1 + (iw - dkw + 2*pad_w) / horizontal_stride; 

       this->output_shape = {static_cast<unsigned int>(on),
                             static_cast<unsigned int>(oc),
                             static_cast<unsigned int>(oh),
                             static_cast<unsigned int>(ow)};

#else
       std::fprintf(stderr, "Error: Conv2dForward::output_shape requires GPU.\n");
       this->output_shape = this->input->get_output_shape();
#endif
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {

       cudnnErrchk(
             cudnnGetConvolution2dForwardOutputDim(
                   this->cudnn_settings.conv_desc,
                   this->input_tensor->get_cudnn_tensor_descriptor(),
                   this->cudnn_settings.filter_desc, &on, &oc, &oh, &ow));

        this->output_shape = {static_cast<unsigned int>(on),
                              static_cast<unsigned int>(oc),
                              static_cast<unsigned int>(oh),
                              static_cast<unsigned int>(ow)};
    }
#endif

    this->output_tensor = new Tensor<T>(this->output_shape, {NONE, {}}, this->mem_type);
#if defined(MAGMADNN_HAVE_CUDA)
    this->output_tensor->set_custream(this->get_custream());
    this->output_tensor->set_cublas_handle(this->get_cublas_handle());
#endif

}

template class Conv2DForwardOp<int>;
template class Conv2DForwardOp<float>;
template class Conv2DForwardOp<double>;

template <typename T>
Conv2DForwardOp<T> *conv2dforward(Operation<T> *input, Operation<T> *filter, int pad_h, int pad_w, int vertical_stride,
                                  int horizontal_stride, int dilation_h, int dilation_w, bool use_cross_correlation,
                                  bool needs_grad) {
    return new Conv2DForwardOp<T>(input, filter, pad_h, pad_w, vertical_stride, horizontal_stride, dilation_h,
                                  dilation_w, use_cross_correlation, needs_grad);
}
template Conv2DForwardOp<int> *conv2dforward(Operation<int> *input, Operation<int> *filter, int pad_h, int pad_w,
                                             int vertical_stride, int horizontal_stride, int dilation_h, int dilation_w,
                                             bool use_cross_correlation, bool needs_grad);
template Conv2DForwardOp<float> *conv2dforward(Operation<float> *input, Operation<float> *filter, int pad_h, int pad_w,
                                               int vertical_stride, int horizontal_stride, int dilation_h,
                                               int dilation_w, bool use_cross_correlation, bool needs_grad);
template Conv2DForwardOp<double> *conv2dforward(Operation<double> *input, Operation<double> *filter, int pad_h,
                                                int pad_w, int vertical_stride, int horizontal_stride, int dilation_h,
                                                int dilation_w, bool use_cross_correlation, bool needs_grad);

}  // namespace op
}  // namespace magmadnn
