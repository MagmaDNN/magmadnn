
#include "compute/batchnorm/batchnormop.h"

#if defined(MAGMADNN_CMAKE_BUILD)
#include "magmadnn/config.h"
#endif

namespace magmadnn {
namespace op {

template <typename T>
BatchNormOp<T>::BatchNormOp(Operation<T> *input, bool needs_grad)
    : Operation<T>::Operation({input}, needs_grad), input(input), num_calls(0) {
    /* setup code in here */
    this->output_shape = input->get_output_shape();
    this->mem_type = input->get_memory_type();
    this->name = "BatchNorm";

    this->input_tensor = input->get_output_tensor();
    this->output_tensor = new Tensor<T>(this->output_shape, {NONE, {}}, this->mem_type);

#if defined(MAGMADNN_HAVE_CUDA)
    init_settings();
#endif
}

template <typename T>
BatchNormOp<T>::~BatchNormOp() {}

template <typename T>
Tensor<T> *BatchNormOp<T>::_eval(bool recompute) {
    /* eval code in here ... */

    input_tensor = input->eval(recompute);

    if (this->mem_type == HOST) {
        math::batchnorm(input_tensor, this->output_tensor);
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {
        this->settings.handle = this->get_cudnn_handle();
        math::batchnorm_device(input_tensor, this->output_tensor, bn_scale, bn_bias, running_mean, running_variance,
                               saved_mean, saved_variance, num_calls, this->settings);
        if (!this->get_async()) cudaStreamSynchronize(this->get_custream());
    }
#endif

    return this->output_tensor;
}

template <typename T>
Tensor<T> *BatchNormOp<T>::_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad) {
    /* return gradient in here ... */
    Tensor<T> *out = this->_grad_cache[(uintptr_t) var];

    if (out == NULL) {
        out = new Tensor<T>(this->output_shape, {NONE, {}}, this->mem_type);
#if defined(MAGMADNN_HAVE_CUDA)
        out->set_custream(this->get_custream());
        out->set_cublas_handle(this->get_cublas_handle());
#endif
        this->_grad_cache[(uintptr_t) var] = out;
    }

    if (this->mem_type == HOST) {
        math::batchnorm_grad(grad, out);
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {
        this->settings.handle = this->get_cudnn_handle();
        math::batchnorm_grad_device(this->input_tensor, grad, out, bn_scale, bn_scale_diff, bn_bias_diff, saved_mean,
                                    saved_variance, this->settings);
        if (!this->get_async()) cudaStreamSynchronize(this->get_custream());
    }
#endif

    return out;
}

#if defined(MAGMADNN_HAVE_CUDA)
template <typename T>
void BatchNormOp<T>::init_settings() {
    settings.handle = ::magmadnn::internal::MAGMADNN_SETTINGS->cudnn_handle;

    /* Use spatial if 4D (conv layer), and use per activation if 2D (fully connected layer) */
    settings.mode = (this->output_shape.size() == 4) ? CUDNN_BATCHNORM_SPATIAL : CUDNN_BATCHNORM_PER_ACTIVATION;
    cudnnErrchk(cudnnCreateTensorDescriptor(&settings.bn_tensor_desc));
    cudnnErrchk(cudnnDeriveBNTensorDescriptor(settings.bn_tensor_desc,
                                              this->input_tensor->get_cudnn_tensor_descriptor(), settings.mode));

    /* Determine and set the dimensions for the normalization tensors */
    std::vector<unsigned int> bn_tensor_shape = this->output_shape;
    bn_tensor_shape[0] = 1;
    if (settings.mode == CUDNN_BATCHNORM_SPATIAL) {
        bn_tensor_shape[2] = 1;
        bn_tensor_shape[3] = 1;
    }

    bn_scale = new Tensor<T>(bn_tensor_shape, {ONE, {}}, this->mem_type);
    bn_bias = new Tensor<T>(bn_tensor_shape, {ONE, {}}, this->mem_type);
    bn_scale_diff = new Tensor<T>(bn_tensor_shape, {ONE, {}}, this->mem_type);
    bn_bias_diff = new Tensor<T>(bn_tensor_shape, {ONE, {}}, this->mem_type);
    running_mean = new Tensor<T>(bn_tensor_shape, {ZERO, {}}, this->mem_type);
    running_variance = new Tensor<T>(bn_tensor_shape, {ZERO, {}}, this->mem_type);
    saved_mean = new Tensor<T>(bn_tensor_shape, {ZERO, {}}, this->mem_type);
    saved_variance = new Tensor<T>(bn_tensor_shape, {ZERO, {}}, this->mem_type);
}

#endif

template class BatchNormOp<int>;
template class BatchNormOp<float>;
template class BatchNormOp<double>;

template <typename T>
BatchNormOp<T> *batchnorm(Operation<T> *input, bool needs_grad) {
    return new BatchNormOp<T>(input, needs_grad);
}
template BatchNormOp<int> *batchnorm(Operation<int> *input, bool needs_grad);
template BatchNormOp<float> *batchnorm(Operation<float> *input, bool needs_grad);
template BatchNormOp<double> *batchnorm(Operation<double> *input, bool needs_grad);

}  // namespace op
}  // namespace magmadnn
