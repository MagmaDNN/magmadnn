#include "compute/conv2dforward/conv2dforwardop.h"

#include <iostream>

#include "magmadnn/config.h"

namespace magmadnn {
namespace op {

template <typename T>
void Conv2DForwardOp<T>::onednn_backward_weights(Tensor<T> *grad, Tensor<T> *out) {

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
}
   
template <typename T>
void Conv2DForwardOp<T>::onednn_backward_data(Tensor<T> *grad, Tensor<T> *out) {

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

}

template <typename T>
void Conv2DForwardOp<T>::onednn_forward() {

   auto src_md = this->dnnl_fwd_pdesc_->src_desc();
   auto dst_md = this->dnnl_fwd_pdesc_->dst_desc();
   auto bias_md = this->dnnl_fwd_pdesc_->bias_desc();
   auto weights_md = this->dnnl_fwd_pdesc_->weights_desc();

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
}

template class Conv2DForwardOp<int>;
template class Conv2DForwardOp<float>;
template class Conv2DForwardOp<double>;
   
}}  // End of namespace magmadnn::op

