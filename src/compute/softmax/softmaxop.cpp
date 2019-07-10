
#include "compute/softmax/softmaxop.h"

namespace magmadnn {
namespace op {

template <typename T>
SoftmaxOp<T>::SoftmaxOp(Operation<T> *input, bool copy, bool needs_grad)
    : Operation<T>::Operation({input}, needs_grad), input(input), copy(copy) {
    /* setup code in here */
    this->output_shape = input->get_output_shape();
    this->mem_type = input->get_memory_type();
    this->name = "Softmax";

    this->input_tensor = input->get_output_tensor();
    this->output_tensor = new Tensor<T>(this->output_shape, {NONE, {}}, this->mem_type);

#if defined(_HAS_CUDA_)
    init_settings();
#endif
}

template <typename T>
Tensor<T> *SoftmaxOp<T>::_eval(bool recompute) {
    /* eval code in here ... */

    input_tensor = input->eval(recompute);

    if (this->mem_type == HOST) {
        math::softmax(input_tensor, this->output_tensor);
    }
#if defined(_HAS_CUDA_)
    else {
        math::softmax_device(input_tensor, this->output_tensor, this->settings);
    }
#endif

    return this->output_tensor;
}

template <typename T>
Tensor<T> *SoftmaxOp<T>::_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad) {
    /* return gradient in here ... */
    Tensor<T> *out = this->_grad_cache[(uintptr_t) var];

#if defined(_HAS_CUDA_)
    this->grad_settings.dydesc = grad->get_cudnn_tensor_descriptor();
#endif

    if (out == NULL) {
        out = new Tensor<T>(this->output_shape, {NONE, {}}, this->mem_type);
        this->_grad_cache[(uintptr_t) var] = out;

#if defined(_HAS_CUDA_)
        this->grad_settings.dxdesc = out->get_cudnn_tensor_descriptor();
#endif
    }

    if (this->mem_type == HOST) {
        math::softmax_grad(this->output_tensor, grad, out);
    }
#if defined(_HAS_CUDA_)
    else {
        math::softmax_grad_device(this->output_tensor, grad, out, this->grad_settings);
    }
#endif

    return out;
}

#if defined(_HAS_CUDA_)
template <typename T>
void SoftmaxOp<T>::init_settings() {
    this->settings.handle = ::magmadnn::internal::MAGMADNN_SETTINGS->cudnn_handle;
    this->settings.alg = CUDNN_SOFTMAX_ACCURATE;
    this->settings.mode = CUDNN_SOFTMAX_MODE_INSTANCE;

    this->settings.xdesc = this->input_tensor->get_cudnn_tensor_descriptor();
    this->settings.ydesc = this->output_tensor->get_cudnn_tensor_descriptor();

    this->grad_settings.handle = ::magmadnn::internal::MAGMADNN_SETTINGS->cudnn_handle;
    this->grad_settings.alg = CUDNN_SOFTMAX_ACCURATE;
    this->grad_settings.mode = CUDNN_SOFTMAX_MODE_INSTANCE;
    this->grad_settings.ydesc = this->output_tensor->get_cudnn_tensor_descriptor();
    /* hold off to init grad tensor descriptors */
}
#endif

template class SoftmaxOp<int>;
template class SoftmaxOp<float>;
template class SoftmaxOp<double>;

template <typename T>
SoftmaxOp<T> *softmax(Operation<T> *input, bool copy, bool needs_grad) {
    return new SoftmaxOp<T>(input, copy, needs_grad);
}
template SoftmaxOp<int> *softmax(Operation<int> *input, bool copy, bool needs_grad);
template SoftmaxOp<float> *softmax(Operation<float> *input, bool copy, bool needs_grad);
template SoftmaxOp<double> *softmax(Operation<double> *input, bool copy, bool needs_grad);

}  // namespace op
}  // namespace magmadnn