
#include "compute/conv2dforward/conv2dforwardop.h"

namespace magmadnn {
namespace op {

Conv2DForwardOp::Conv2DForwardOp(Operation *input, Operation *filter, int pad_h, int pad_w, int vertical_stride,
                                 int horizontal_stride, int dilation_h, int dilation_w, bool use_cross_correlation,
                                 bool needs_grad)
    : Operation::Operation({input, filter}, needs_grad),
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
    this->mem_type_ = input->get_memory_type();

    /* initialize all the conv settings */
    this->dtype_ = this->input->dtype();
    this->init_settings();
}

Conv2DForwardOp::~Conv2DForwardOp() {
    if (this->mem_type_ == HOST) {
        /* delete CPU workspace here */
    }
#if defined(_HAS_CUDA_)
    else {

        cudaErrchk(cudaFree(this->cudnn_settings.workspace));
        cudaErrchk(cudaFree(this->cudnn_settings.grad_data_workspace));
        cudaErrchk(cudaFree(this->cudnn_settings.grad_filter_workspace));

        cudnnErrchk(cudnnDestroyFilterDescriptor(this->cudnn_settings.filter_desc));
        cudnnErrchk(cudnnDestroyConvolutionDescriptor(this->cudnn_settings.conv_desc));
    }
#endif
}

Tensor &Conv2DForwardOp::_eval(bool recompute) {
    Tensor &input_tensor = input->eval(recompute);
    Tensor &filter_tensor = filter->eval(recompute);

    switch (getDeviceType(this->mem_type_)) {
        case CPU:
            ::magmadnn::math::conv2d<CPU>(input_tensor, filter_tensor, this->output_tensor_);
#if defined(_HAS_CUDA_)
        case GPU: /* only wrap this in a macro, because cudnn_settings won't be defined on host */
            ::magmadnn::math::conv2d<GPU>(input_tensor, filter_tensor, this->output_tensor_, this->cudnn_settings);
#endif
        default:
            ::magmadnn::math::conv2d<CPU>(input_tensor, filter_tensor, this->output_tensor_);
    }

    return this->output_tensor_;
}

Tensor &Conv2DForwardOp::_grad(Operation *consumer, Operation *var, const Tensor &grad) {
    /* return gradient in here ... */

    auto ret = this->_grad_cache.find(var);
    Tensor out;

    if (var == this->input) {
        if (!ret->first) {
            out = Tensor(this->input->get_output_shape(), this->dtype_, {NONE, {}}, this->mem_type_);
            this->_grad_cache.insert(std::make_pair(var, out));
        }

        Tensor &filter_tensor = this->filter->eval(false);

        switch (getDeviceType(this->mem_type_)) {
            case CPU:
                ::magmadnn::math::conv2d_grad_data<CPU>(filter_tensor, grad, out);
#if defined(_HAS_CUDA_)
            case GPU:
                ::magmadnn::math::conv2d_grad_data<GPU>(filter_tensor, grad, out, this->cudnn_settings);
#endif
            default:
                LOG(ERROR) << "Unsupported conv2D type.\n";
        }

    } else if (var == this->filter) {
        if (!ret->first) {
            out = Tensor(this->filter->get_output_shape(), this->dtype_, {NONE, {}}, this->mem_type_);
            this->_grad_cache.insert(std::make_pair(var, out));
        }

        Tensor &input_tensor = this->input->eval(false);

        switch (getDeviceType(this->mem_type_)) {
            case CPU:
                ::magmadnn::math::conv2d_grad_filter<CPU>(input_tensor, grad, out);
#if defined(_HAS_CUDA_)
            case GPU:
                ::magmadnn::math::conv2d_grad_filter<GPU>(input_tensor, grad, out, this->cudnn_settings);
#endif
            default:
                LOG(ERROR) << "Unsupported conv2D type.\n";
        }

    } else {
        std::fprintf(stderr, "Error: bad conv2d grad\n");
    }

    return this->_grad_cache[var];
}

void Conv2DForwardOp::init_settings() {
    if (this->mem_type_ == HOST) {
        std::fprintf(stderr, "Error: Conv2DForward::init_settings requires GPU.\n");
    }
#if defined(_HAS_CUDA_)
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
        cudaErrchk(cudaMalloc((void **) &this->cudnn_settings.grad_data_workspace,
                              this->cudnn_settings.grad_data_workspace_size));

        cudnnErrchk(cudnnGetConvolutionBackwardFilterWorkspaceSize(
            ::magmadnn::internal::MAGMADNN_SETTINGS->cudnn_handle, this->input_tensor->get_cudnn_tensor_descriptor(),
            this->output_tensor->get_cudnn_tensor_descriptor(), this->cudnn_settings.conv_desc,
            this->cudnn_settings.filter_desc, this->cudnn_settings.bwd_filter_algo,
            &this->cudnn_settings.grad_filter_workspace_size));
        cudaErrchk(cudaMalloc((void **) &this->cudnn_settings.grad_filter_workspace,
                              this->cudnn_settings.grad_filter_workspace_size));
    }
#endif
}

void Conv2DForwardOp::calculate_and_set_output_shape() {
    /* calculate the correct output shape here */
    if (this->mem_type_ == HOST) {
        std::fprintf(stderr, "Error: Conv2dForward::output_shape requires GPU.\n");
        this->output_shape_ = this->input->get_output_shape();
    }
#if defined(_HAS_CUDA_)
    else {
        int n, c, h, w;

        cudnnErrchk(cudnnGetConvolution2dForwardOutputDim(this->cudnn_settings.conv_desc,
                                                          this->input_tensor->get_cudnn_tensor_descriptor(),
                                                          this->cudnn_settings.filter_desc, &n, &c, &h, &w));
        this->output_shape_ = {static_cast<unsigned int>(n), static_cast<unsigned int>(c), static_cast<unsigned int>(h),
                               static_cast<unsigned int>(w)};
    }
#endif

    this->output_tensor_ = Tensor(this->output_shape_, this->dtype_, {NONE, {}}, this->mem_type_);
}

}  // namespace op
}  // namespace magmadnn