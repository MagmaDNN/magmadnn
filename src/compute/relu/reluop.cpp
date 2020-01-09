/**
 * @file reluop.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-01
 *
 * @copyright Copyright (c) 2019
 */
#include "compute/relu/reluop.h"

namespace magmadnn {
namespace op {

template <typename T>
ReluOp<T>::ReluOp(Operation<T> *x, bool copy, bool needs_grad)
    : Operation<T>::Operation({x}, needs_grad), x(x), copy(copy) {
    this->output_shape = x->get_output_shape();
    this->mem_type = x->get_memory_type();

    if (copy) {
        this->output_tensor = new Tensor<T>(this->output_shape, this->mem_type);
    } else {
        fprintf(stderr, "inplace relu not defined\n");
    }

#if defined(MAGMADNN_HAVE_CUDA)
    cudnnErrchk(cudnnCreateActivationDescriptor(&cudnn_settings.descriptor));
    cudnnErrchk(
        cudnnSetActivationDescriptor(
              cudnn_settings.descriptor,
              CUDNN_ACTIVATION_RELU,
              CUDNN_NOT_PROPAGATE_NAN,
              1.0));
#endif
}

template <typename T>
ReluOp<T>::~ReluOp() {
#if defined(MAGMADNN_HAVE_CUDA)
    cudnnErrchk(cudnnDestroyActivationDescriptor(cudnn_settings.descriptor));
#endif
}

template <typename T>
Tensor<T> *ReluOp<T>::_eval(bool recompute) {
    x_tensor = x->eval(recompute);

    this->output_tensor = x_tensor;

    // internal::relu_full(x_tensor, this->output_tensor);
    if (this->mem_type == HOST) {
        math::relu(x_tensor, this->output_tensor);
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {
       this->cudnn_settings.handle = this->get_cudnn_handle();
       math::relu_device(x_tensor, this->output_tensor, this->cudnn_settings);
       if (!this->get_async()) cudaStreamSynchronize(this->get_custream());
    }
#endif

    return this->output_tensor;
}

template <typename T>
Tensor<T> *ReluOp<T>::_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad) {
    Tensor<T> *out = this->_grad_cache[(uintptr_t) var];

    if (out == NULL) {
        out = new Tensor<T>(this->output_shape, {NONE, {}}, this->mem_type);
#if defined(MAGMADNN_HAVE_CUDA)
            out->set_custream(this->get_custream());
            out->set_cublas_handle(this->get_cublas_handle());
#endif
        this->_grad_cache[(uintptr_t) var] = out;
    }

    x_tensor = x->eval(false);
    if (this->mem_type == HOST) {
        math::relu_grad(this->x_tensor, this->output_tensor, grad, out);
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {
       this->cudnn_settings.handle = this->get_cudnn_handle();
       math::relu_grad_device(
             x_tensor, this->output_tensor, grad, out,
             this->cudnn_settings);
       if (!this->get_async()) cudaStreamSynchronize(this->get_custream());
    }
#endif

    return out;
}

template class ReluOp<int>;
template class ReluOp<float>;
template class ReluOp<double>;

template <typename T>
ReluOp<T> *relu(Operation<T> *x, bool copy, bool needs_grad) {
    return new ReluOp<T>(x, copy, needs_grad);
}
template ReluOp<int> *relu(Operation<int> *x, bool copy, bool needs_grad);
template ReluOp<float> *relu(Operation<float> *x, bool copy, bool needs_grad);
template ReluOp<double> *relu(Operation<double> *x, bool copy, bool needs_grad);

}  // namespace op
}  // namespace magmadnn
