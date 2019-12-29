#include <iostream>

#include "compute/conv2dforward/conv2dforwardop.h"

namespace magmadnn {
namespace op {

template <typename T>
Conv2DForwardOp<T>::Conv2DForwardOp(Operation<T> *input, Operation<T> *filter, int pad_h, int pad_w,
                                    int vertical_stride, int horizontal_stride, int dilation_h, int dilation_w,
                                    bool use_cross_correlation, bool needs_grad)
    : Operation<T>::Operation({input, filter}, needs_grad),
      input(input),
      filter(filter),
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
    input_tensor = input->eval(recompute);
    filter_tensor = filter->eval(recompute);

    if (this->mem_type == HOST) {
        std::fprintf(stderr, "Error: Conv2dForward::_eval requires GPU\n");
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {
        ::magmadnn::math::conv2d_device(this->input_tensor, this->filter_tensor, this->output_tensor,
                                        this->cudnn_settings);
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
            this->_grad_cache[(uintptr_t) var] = out;
        }

        this->filter_tensor = this->filter->eval(false);

        if (this->mem_type == HOST) {
            ::magmadnn::math::conv2d_grad_data(this->filter_tensor, grad, out);
        }
#if defined(MAGMADNN_HAVE_CUDA)
        else {
            ::magmadnn::math::conv2d_grad_data_device(this->filter_tensor, grad, out, this->cudnn_settings);
        }
#endif

    } else if (var == this->filter) {
        if (out == NULL) {
            out = new Tensor<T>(this->filter->get_output_shape(), {NONE, {}}, this->mem_type);
            this->_grad_cache[(uintptr_t) var] = out;
        }

        this->input_tensor = this->input->eval(false);

        if (this->mem_type == HOST) {
            ::magmadnn::math::conv2d_grad_filter(this->input_tensor, grad, out);
        }
#if defined(MAGMADNN_HAVE_CUDA)
        else {
            ::magmadnn::math::conv2d_grad_filter_device(this->input_tensor, grad, out, this->cudnn_settings);
        }
#endif

    } else {
        std::fprintf(stderr, "Error: bad conv2d grad\n");
    }

    return out;
}

template <typename T>
void Conv2DForwardOp<T>::init_settings() {
    if (this->mem_type == HOST) {
        std::fprintf(stderr, "Error: Conv2DForward::init_settings requires GPU.\n");
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {
        /* init the conv descriptor */
        cudnnErrchk(cudnnCreateConvolutionDescriptor(&this->cudnn_settings.conv_desc));

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
            ::magmadnn::internal::MAGMADNN_SETTINGS->cudnn_handle,
            this->input->get_output_tensor()->get_cudnn_tensor_descriptor(), this->cudnn_settings.filter_desc,
            this->cudnn_settings.conv_desc, this->output_tensor->get_cudnn_tensor_descriptor(),
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &this->cudnn_settings.algo));

        /* use CuDNN to get the necessary workspace size and allocate that memory */
        cudnnErrchk(cudnnGetConvolutionForwardWorkspaceSize(
            ::magmadnn::internal::MAGMADNN_SETTINGS->cudnn_handle,
            this->input->get_output_tensor()->get_cudnn_tensor_descriptor(), this->cudnn_settings.filter_desc,
            this->cudnn_settings.conv_desc, this->output_tensor->get_cudnn_tensor_descriptor(),
            this->cudnn_settings.algo, &this->cudnn_settings.workspace_size));
        // std::cout << "Conv2DForwardOp<T>::init_settings, forward workspace size (MB) = "
        //           << (float) ((float) this->cudnn_settings.workspace_size / ((float) 1024.0 * 1024.0))  << std::endl;
        cudaErrchk(cudaMalloc((void **) &this->cudnn_settings.workspace, this->cudnn_settings.workspace_size));

        /* INIT the grad settings */
        cudnnErrchk(cudnnGetConvolutionBackwardDataAlgorithm(
            ::magmadnn::internal::MAGMADNN_SETTINGS->cudnn_handle, this->cudnn_settings.filter_desc,
            this->output_tensor->get_cudnn_tensor_descriptor(),                                /* use output for dy */
            this->cudnn_settings.conv_desc, this->input_tensor->get_cudnn_tensor_descriptor(), /* use input for dx */
            CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &this->cudnn_settings.bwd_data_algo));

        cudnnErrchk(cudnnGetConvolutionBackwardFilterAlgorithm(
            ::magmadnn::internal::MAGMADNN_SETTINGS->cudnn_handle, this->input_tensor->get_cudnn_tensor_descriptor(),
            this->output_tensor->get_cudnn_tensor_descriptor(), this->cudnn_settings.conv_desc,
            this->cudnn_settings.filter_desc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0,
            &this->cudnn_settings.bwd_filter_algo));

        /* get the workspaces for each of the backward algorithms */
        cudnnErrchk(cudnnGetConvolutionBackwardDataWorkspaceSize(
            ::magmadnn::internal::MAGMADNN_SETTINGS->cudnn_handle, this->cudnn_settings.filter_desc,
            this->output_tensor->get_cudnn_tensor_descriptor(), this->cudnn_settings.conv_desc,
            this->input_tensor->get_cudnn_tensor_descriptor(), this->cudnn_settings.bwd_data_algo,
            &this->cudnn_settings.grad_data_workspace_size));
        // std::cout << "Conv2DForwardOp<T>::init_settings, backward workspace size (MB) = "
        //           << (float) ((float) this->cudnn_settings.grad_data_workspace_size / ((float) 1024.0 * 1024.0))
        //           << std::endl;
        cudaErrchk(cudaMalloc((void **) &this->cudnn_settings.grad_data_workspace,
                              this->cudnn_settings.grad_data_workspace_size));

        cudnnErrchk(cudnnGetConvolutionBackwardFilterWorkspaceSize(
            ::magmadnn::internal::MAGMADNN_SETTINGS->cudnn_handle, this->input_tensor->get_cudnn_tensor_descriptor(),
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
    /* calculate the correct output shape here */
    if (this->mem_type == HOST) {
        std::fprintf(stderr, "Error: Conv2dForward::output_shape requires GPU.\n");
        this->output_shape = this->input->get_output_shape();
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {
        int n, c, h, w;

        cudnnErrchk(cudnnGetConvolution2dForwardOutputDim(this->cudnn_settings.conv_desc,
                                                          this->input_tensor->get_cudnn_tensor_descriptor(),
                                                          this->cudnn_settings.filter_desc, &n, &c, &h, &w));
        this->output_shape = {static_cast<unsigned int>(n), static_cast<unsigned int>(c), static_cast<unsigned int>(h),
                              static_cast<unsigned int>(w)};
    }
#endif

    this->output_tensor = new Tensor<T>(this->output_shape, {NONE, {}}, this->mem_type);
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
