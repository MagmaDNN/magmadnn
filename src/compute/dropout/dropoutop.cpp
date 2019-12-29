
#include "compute/dropout/dropoutop.h"

namespace magmadnn {
namespace op {

template <typename T>
DropoutOp<T>::DropoutOp(Operation<T> *input, float dropout_rate, unsigned long long seed, bool copy, bool needs_grad)
    : Operation<T>::Operation({input}, needs_grad), input(input), dropout_rate(dropout_rate), seed(seed), copy(copy) {
    /* setup code in here */
    assert(dropout_rate >= 0 && dropout_rate <= 1);

    this->output_shape = input->get_output_shape();
    this->mem_type = input->get_memory_type();
    this->name = "Dropout";

    this->input_tensor = input->get_output_tensor();
    this->output_tensor = new Tensor<T>(this->output_shape, {NONE, {}}, this->mem_type);

    T p = 1.0f - dropout_rate;
    this->mask_tensor =
        new Tensor<T>(this->output_shape, {MASK, {static_cast<T>(p), static_cast<T>(1.0f / p)}}, this->mem_type);

#if defined(MAGMADNN_HAVE_CUDA)
    init_settings();
#endif
}

template <typename T>
DropoutOp<T>::~DropoutOp() {
    if (mask_tensor != NULL) delete mask_tensor;

#if defined(MAGMADNN_HAVE_CUDA)
    cudnnErrchk(cudnnDestroyDropoutDescriptor(shared_settings.dropoutDesc));
#endif
}

template <typename T>
Tensor<T> *DropoutOp<T>::_eval(bool recompute) {
    /* eval code in here ... */

    input_tensor = input->eval(recompute);

    if (this->mem_type == HOST) {
        math::dropout(input_tensor, this->output_tensor, mask_tensor, dropout_rate);
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {
        math::dropout_device(input_tensor, this->output_tensor, this->settings, this->shared_settings);
    }
#endif

    return this->output_tensor;
}

template <typename T>
Tensor<T> *DropoutOp<T>::_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad) {
    /* return gradient in here ... */
    Tensor<T> *out = this->_grad_cache[(uintptr_t) var];

#if defined(MAGMADNN_HAVE_CUDA)
    this->grad_settings.dydesc = grad->get_cudnn_tensor_descriptor();
#endif

    if (out == NULL) {
        out = new Tensor<T>(this->output_shape, {NONE, {}}, this->mem_type);
        this->_grad_cache[(uintptr_t) var] = out;

#if defined(MAGMADNN_HAVE_CUDA)
        this->grad_settings.dxdesc = out->get_cudnn_tensor_descriptor();
#endif
    }

    if (this->mem_type == HOST) {
        math::dropout_grad(grad, out, mask_tensor);
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {
        math::dropout_grad_device(grad, out, this->grad_settings, this->shared_settings);
    }
#endif

    return out;
}

#if defined(MAGMADNN_HAVE_CUDA)
template <typename T>
void DropoutOp<T>::init_settings() {
    settings.xdesc = this->input_tensor->get_cudnn_tensor_descriptor();
    settings.ydesc = this->output_tensor->get_cudnn_tensor_descriptor();
    settings.handle = ::magmadnn::internal::MAGMADNN_SETTINGS->cudnn_handle;

    cudnnErrchk(cudnnDropoutGetStatesSize(settings.handle, &shared_settings.stateSizeInBytes));
    cudnnErrchk(cudnnDropoutGetReserveSpaceSize(settings.xdesc, &shared_settings.reserveSpaceSizeInBytes));
    cudaErrchk(cudaMalloc(&shared_settings.states, shared_settings.stateSizeInBytes));
    cudaErrchk(cudaMalloc(&shared_settings.reserveSpace, shared_settings.reserveSpaceSizeInBytes));

    cudnnErrchk(cudnnCreateDropoutDescriptor(&shared_settings.dropoutDesc));
    cudnnErrchk(cudnnSetDropoutDescriptor(shared_settings.dropoutDesc, settings.handle, this->dropout_rate,
                                          shared_settings.states, shared_settings.stateSizeInBytes, this->seed));

    grad_settings.handle = ::magmadnn::internal::MAGMADNN_SETTINGS->cudnn_handle;
    /* hold off to init grad tensor descriptors */
}

#endif

template class DropoutOp<int>;
template class DropoutOp<float>;
template class DropoutOp<double>;

template <typename T>
DropoutOp<T> *dropout(Operation<T> *input, float dropout_rate, unsigned long long seed, bool copy, bool needs_grad) {
    return new DropoutOp<T>(input, dropout_rate, seed, copy, needs_grad);
}
template DropoutOp<int> *dropout(Operation<int> *input, float dropout_rate, unsigned long long seed, bool copy,
                                 bool needs_grad);
template DropoutOp<float> *dropout(Operation<float> *input, float dropout_rate, unsigned long long seed, bool copy,
                                   bool needs_grad);
template DropoutOp<double> *dropout(Operation<double> *input, float dropout_rate, unsigned long long seed, bool copy,
                                    bool needs_grad);

}  // namespace op
}  // namespace magmadnn
