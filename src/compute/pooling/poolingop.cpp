
#include "compute/pooling/poolingop.h"

namespace magmadnn {
namespace op {

template <typename T>
PoolingOp<T>::PoolingOp(Operation<T> *input, int filter_h, int filter_w, int pad_h, int pad_w, int vertical_stride,
                        int horizontal_stride, pooling_mode mode, bool propagate_nan, bool needs_grad)
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
        ::magmadnn::math::pooling_device(this->input_tensor, this->output_tensor, this->settings);
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
        this->_grad_cache[(uintptr_t) var] = out;
    }

    if (this->mem_type == HOST) {
        ::magmadnn::math::pooling_grad(this->input_tensor, this->output_tensor, grad, out);
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {
        ::magmadnn::math::pooling_grad_device(this->input_tensor, this->output_tensor, grad, out, this->settings);
    }
#endif

    return out;
}

template <typename T>
void PoolingOp<T>::init_settings() {
    if (this->mem_type == HOST) {
        std::fprintf(stderr, "Error: PoolingOp::init_settings requires GPU.\n");
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {
        /* init the pooling descriptor */
        cudnnErrchk(cudnnCreatePoolingDescriptor(&this->settings.poolingDesc));

        /* set the pooling description */
        cudnnErrchk(cudnnSetPooling2dDescriptor(
            this->settings.poolingDesc,
            (mode == MAX_POOL) ? CUDNN_POOLING_MAX_DETERMINISTIC : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
            (propagate_nan) ? CUDNN_PROPAGATE_NAN : CUDNN_NOT_PROPAGATE_NAN, filter_h, filter_w, pad_h, pad_w,
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
        int n, c, h, w;

        cudnnErrchk(cudnnGetPooling2dForwardOutputDim(
            this->settings.poolingDesc, this->input_tensor->get_cudnn_tensor_descriptor(), &n, &c, &h, &w));
        this->output_shape = {static_cast<unsigned int>(n), static_cast<unsigned int>(c), static_cast<unsigned int>(h),
                              static_cast<unsigned int>(w)};
    }
#endif

    this->output_tensor = new Tensor<T>(this->output_shape, {NONE, {}}, this->mem_type);
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
