
#include "compute/batchnorm/batchnormop.h"

namespace magmadnn {
namespace op {

BatchNormOp::BatchNormOp(Operation *input) : Operation::Operation({input}), input(input), num_calls(0) {
    /* setup code in here */
    this->use_operation_settings(input, true);
    this->name_ = "BatchNorm";

    this->output_tensor_ = Tensor(this->output_shape_, this->dtype_, {NONE}, this->mem_type_);

#if defined(_HAS_CUDA_)
    init_settings();
#endif
}

Tensor &BatchNormOp::_eval(bool recompute) {
    /* eval code in here ... */

    Tensor &input_tensor = input->eval(recompute);

    if (this->mem_type_ == HOST) {
        math::batchnorm(input_tensor, this->output_tensor_);
    }
#if defined(_HAS_CUDA_)
    else {
        math::batchnorm_device(input_tensor, this->output_tensor_, bn_scale, bn_bias, running_mean, running_variance,
                               saved_mean, saved_variance, num_calls, this->settings);
    }
#endif

    return this->output_tensor_;
}

Tensor &BatchNormOp::_grad(Operation *consumer, Operation *var, const Tensor &grad) {
    /* return gradient in here ... */
    auto res = this->_grad_cache.find(var);
    Tensor out;

    if (!res->first) {
        out = Tensor(this->output_shape_, this->dtype_, {NONE, {}}, this->mem_type_);
        this->_grad_cache.insert(std::make_pair(var, std::ref(out)));
    }

    if (this->mem_type_ == HOST) {
        math::batchnorm_grad(grad, out);
    }
#if defined(_HAS_CUDA_)
    else {
        math::batchnorm_grad_device(this->input_tensor, grad, out, bn_scale, bn_scale_diff, bn_bias_diff, saved_mean,
                                    saved_variance, this->settings);
    }
#endif

    return this->_grad_cache[var];
}

#if defined(_HAS_CUDA_)
void BatchNormOp::init_settings() {
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

    bn_scale = Tensor(bn_tensor_shape, {ONE, {}}, this->mem_type);
    bn_bias = Tensor(bn_tensor_shape, {ONE, {}}, this->mem_type);
    bn_scale_diff = Tensor(bn_tensor_shape, {ONE, {}}, this->mem_type);
    bn_bias_diff = Tensor(bn_tensor_shape, {ONE, {}}, this->mem_type);
    running_mean = Tensor(bn_tensor_shape, {ZERO, {}}, this->mem_type);
    running_variance = Tensor(bn_tensor_shape, {ZERO, {}}, this->mem_type);
    saved_mean = Tensor(bn_tensor_shape, {ZERO, {}}, this->mem_type);
    saved_variance = Tensor(bn_tensor_shape, {ZERO, {}}, this->mem_type);
}

#endif

}  // namespace op
}  // namespace magmadnn