#include "compute/pooling/poolingop.h"

#include "magmadnn/config.h"

namespace magmadnn {
namespace op {

template <typename T>
PoolingOp<T>::PoolingOp(Operation<T> *input, int filter_h, int filter_w, int pad_h, int pad_w, int vertical_stride,
                        int horizontal_stride, int dilation_h, int dilation_w, pooling_mode mode, bool propagate_nan,
                        bool needs_grad)
    : Operation<T>::Operation({input}, needs_grad),
      input(input),
      filter_h(filter_h),
      filter_w(filter_w),
      pad_h(pad_h),
      pad_w(pad_w),
      vertical_stride(vertical_stride),
      horizontal_stride(horizontal_stride),
      dilation_h(dilation_h),
      dilation_w(dilation_w),
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
        if (this->mode == pooling_mode::MAX_POOL)
            this->max_positions = new Tensor<int>(this->output_shape, {ZERO, {}}, this->mem_type);

        ::magmadnn::math::pooling(this->input_tensor, this->output_tensor, this->max_positions, this->filter_h,
                                  this->filter_w, this->pad_h, this->pad_w, this->vertical_stride,
                                  this->horizontal_stride, this->dilation_h, this->dilation_w, this->mode);
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {
        this->settings.handle = this->get_cudnn_handle();
        if (this->dilation_h != 1 || this->dilation_w != 1)
            fprintf(stderr, "Warning: PoolingOp::pooling_device does not support dilation\n");
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
        ::magmadnn::math::pooling_grad(this->input_tensor, this->output_tensor, grad, this->max_positions, out,
                                       this->filter_h, this->filter_w, this->pad_h, this->pad_w, this->vertical_stride,
                                       this->horizontal_stride, this->dilation_h, this->dilation_w, this->mode);
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
#if defined(MAGMADNN_HAVE_CUDA)
    if (this->mem_type != HOST) {
        this->settings.handle = this->get_cudnn_handle();

        /* init the pooling descriptor */
        cudnnErrchk(cudnnCreatePoolingDescriptor(&this->settings.poolingDesc));

        /* set the pooling description */
        cudnnErrchk(cudnnSetPooling2dDescriptor(
            this->settings.poolingDesc,
            (mode == MAX_POOL) ? CUDNN_POOLING_MAX_DETERMINISTIC : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
            (propagate_nan) ? CUDNN_PROPAGATE_NAN : CUDNN_NOT_PROPAGATE_NAN, filter_h, filter_w, pad_h, pad_w,
            vertical_stride, horizontal_stride));
    }
#endif

    this->calculate_and_set_output_shape();
}

template <typename T>
void PoolingOp<T>::calculate_and_set_output_shape() {
    /* calculate the correct output shape here */
    if (this->mem_type == HOST) {
        unsigned int No, Co, Ho, Wo;         // shorthand for tensor dims
        const int p_h_x2 = this->pad_h * 2;  // total padding to add on top and bottom
        const int p_w_x2 = this->pad_w * 2;  // total padding to add on left and right
        std::vector<unsigned int> in_shape = this->input_tensor->get_shape();

        if (in_shape.size() == 4) {
            Ho = (in_shape[2] - (this->filter_h * this->dilation_h) + p_h_x2 + this->vertical_stride) /
                 this->vertical_stride;
            Wo = (in_shape[3] - (this->filter_w * this->dilation_w) + p_w_x2 + this->horizontal_stride) /
                 this->horizontal_stride;
            Co = in_shape[1];
            No = in_shape[0];

            this->output_shape = {No, Co, Ho, Wo};
        } else if (in_shape.size() == 3) {
            Ho = (in_shape[1] - (this->filter_h * this->dilation_h) + p_h_x2 + this->vertical_stride) /
                 this->vertical_stride;
            Wo = (in_shape[2] - (this->filter_w * this->dilation_w) + p_w_x2 + this->horizontal_stride) /
                 this->horizontal_stride;
            Co = in_shape[0];
            this->output_shape = {Co, Ho, Wo};
        } else if (in_shape.size() == 2) {
            Ho = (in_shape[2] - (this->filter_h * this->dilation_h) + p_h_x2 + this->vertical_stride) /
                 this->vertical_stride;
            Wo = (in_shape[3] - (this->filter_w * this->dilation_w) + p_w_x2 + this->horizontal_stride) /
                 this->horizontal_stride;
            this->output_shape = {Ho, Wo};
        } else
            fprintf(stderr, "Error: PoolingOp::invalid shape sizes. In shape: %i \n", in_shape.size());

    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {
        int n, c, h, w;

        cudnnErrchk(cudnnGetPooling2dForwardOutputDim(
            this->settings.poolingDesc, this->input_tensor->get_cudnn_tensor_descriptor(), &n, &c, &h, &w));

        this->output_shape = {static_cast<unsigned int>(n), static_cast<unsigned int>(c), static_cast<unsigned int>(h),
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
                      int horizontal_stride, int dilation_h, int dilation_w, pooling_mode mode, bool propagate_nan,
                      bool needs_grad) {
    return new PoolingOp<T>(input, filter_h, filter_w, pad_h, pad_w, vertical_stride, horizontal_stride, dilation_h,
                            dilation_w, mode, propagate_nan, needs_grad);
}
template PoolingOp<int> *pooling(Operation<int> *input, int filter_h, int filter_w, int pad_h, int pad_w,
                                 int vertical_stride, int horizontal_stride, int dilation_h, int dilation_w,
                                 pooling_mode mode, bool propagate_nan, bool needs_grad);
template PoolingOp<float> *pooling(Operation<float> *input, int filter_h, int filter_w, int pad_h, int pad_w,
                                   int vertical_stride, int horizontal_stride, int dilation_h, int dilation_w,
                                   pooling_mode mode, bool propagate_nan, bool needs_grad);
template PoolingOp<double> *pooling(Operation<double> *input, int filter_h, int filter_w, int pad_h, int pad_w,
                                    int vertical_stride, int horizontal_stride, int dilation_h, int dilation_w,
                                    pooling_mode mode, bool propagate_nan, bool needs_grad);

}  // namespace op
}  // namespace magmadnn
