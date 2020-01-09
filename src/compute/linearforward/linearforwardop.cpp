#include "compute/linearforward/linearforwardop.h"

#include "magmadnn/config.h"

namespace magmadnn {
namespace op {

template <typename T>
LinearForwardOp<T>::LinearForwardOp(Operation<T> *input, Operation<T> *weights, bool copy, bool needs_grad)
    : Operation<T>::Operation({input, weights}, needs_grad),
      input(input),
      weights(weights),
      copy(copy),
      use_bias(false) {
    /* setup code in here */
    this->output_shape = {input->get_output_shape(0), weights->get_output_shape(1)};
    this->mem_type = input->get_memory_type();
    this->name = "LinearForward";

    this->output_tensor = new Tensor<T>(this->output_shape, {NONE, {}}, this->mem_type);
}

template <typename T>
LinearForwardOp<T>::LinearForwardOp(Operation<T> *input, Operation<T> *weights, Operation<T> *bias, bool copy,
                                    bool needs_grad)
    : Operation<T>::Operation({input, weights, bias}, needs_grad),
      input(input),
      weights(weights),
      bias(bias),
      copy(copy),
      use_bias(true) {
    /* setup code in here */
    this->output_shape = {input->get_output_shape(0), weights->get_output_shape(1)};
    this->mem_type = input->get_memory_type();
    this->name = "LinearForward";

    this->output_tensor = new Tensor<T>(this->output_shape, {NONE, {}}, this->mem_type);

    init_bias_settings();
}

template <typename T>
LinearForwardOp<T>::~LinearForwardOp() {
    if (bias_ones != NULL) delete bias_ones;

#if defined(MAGMADNN_HAVE_CUDA)
    cudnnErrchk(cudnnDestroyReduceTensorDescriptor(bias_reduce_settings.descriptor));
    cudaErrchk(cudaFree(bias_reduce_settings.workspace));
#endif
}

template <typename T>
Tensor<T> *LinearForwardOp<T>::_eval(bool recompute) {

    input_tensor = input->eval(recompute);
    weights_tensor = weights->eval(recompute);

    /* XW */
    math::matmul((T) 1, false, input_tensor, false, weights_tensor, (T) 0, this->output_tensor);
#if defined(MAGMADNN_HAVE_CUDA)
    if (!this->get_async()) cudaStreamSynchronize(this->get_custream());
#endif

    if (use_bias) {
        bias_tensor = bias->eval(recompute);
        math::bias_add(this->output_tensor, this->bias_tensor, this->output_tensor);
    }

    return this->output_tensor;
}

template <typename T>
Tensor<T> *LinearForwardOp<T>::_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad) {
    /* wrt input : GW^T  --  wrt weights : X^TG */
    Tensor<T> *out = this->_grad_cache[(uintptr_t) var];

    if (var == this->input) {
        this->weights_tensor = this->weights->eval(false);

        if (out == NULL) {
            out = new Tensor<T>({grad->get_shape(0), this->weights_tensor->get_shape(0)}, {NONE, {}}, this->mem_type);
#if defined(MAGMADNN_HAVE_CUDA)
            out->set_custream(this->get_custream());
            out->set_cublas_handle(this->get_cublas_handle());
#endif
            this->_grad_cache[(uintptr_t) var] = out;
        }

        math::matmul((T) 1, false, grad, true, this->weights_tensor, (T) 0, out);
#if defined(MAGMADNN_HAVE_CUDA)
        if (!this->get_async()) cudaStreamSynchronize(this->get_custream());
#endif

    } else if (var == this->weights) {
        this->input_tensor = this->input->eval(false);

        if (out == NULL) {
            out = new Tensor<T>({this->input_tensor->get_shape(1), grad->get_shape(1)}, {NONE, {}}, this->mem_type);
#if defined(MAGMADNN_HAVE_CUDA)
            out->set_custream(this->get_custream());
            out->set_cublas_handle(this->get_cublas_handle());
#endif
            this->_grad_cache[(uintptr_t) var] = out;
        }

        math::matmul((T) 1, true, this->input_tensor, false, grad, (T) 0, out);
#if defined(MAGMADNN_HAVE_CUDA)
        if (!this->get_async()) cudaStreamSynchronize(this->get_custream());
#endif

    } else if (this->use_bias && var == this->bias) {
        /* grad wrt bias is reduce sum of grad along axis 1 */

        if (out == NULL) {
            out = new Tensor<T>(this->bias_tensor->get_shape(), {NONE, {}}, this->mem_type);
#if defined(MAGMADNN_HAVE_CUDA)
            out->set_custream(this->get_custream());
            out->set_cublas_handle(this->get_cublas_handle());
#endif
            this->_grad_cache[(uintptr_t) var] = out;
        }

        if (this->mem_type == HOST) {
            math::reduce_sum(grad, 1, this->bias_ones, out);
        }
#if defined(MAGMADNN_HAVE_CUDA)
        else {
            this->bias_reduce_settings.cudnn_handle = this->get_cudnn_handle();
            math::reduce_sum_device(grad, 1, out, this->bias_reduce_settings);
            if (!this->get_async()) cudaStreamSynchronize(this->get_custream());
        }
#endif
    }

    return out;
}

template <typename T>
void LinearForwardOp<T>::init_bias_settings() {
    if (this->mem_type == HOST) {
        bias_ones = new Tensor<T>({this->output_shape[1]}, {ONE, {}}, this->mem_type);
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {
        /* create a temporary descriptor for grad, since we do not have its tensor yet and
            therefore cannot call get_cudnn_tensor_descriptor(). This allows us to get the
            workspace size from CuDNN here in the constructor, rather than in eval. */
        cudnnTensorDescriptor_t grad_tmp_descriptor;

        cudnnErrchk(cudnnCreateTensorDescriptor(&grad_tmp_descriptor));
        cudnnErrchk(cudnnSetTensor4dDescriptor(grad_tmp_descriptor, CUDNN_TENSOR_NCHW,
                                               ::magmadnn::internal::get_cudnn_data_type((T) 0),
                                               input->get_output_shape(0), weights->get_output_shape(1), 1, 1));

        cudnnErrchk(cudnnCreateReduceTensorDescriptor(&bias_reduce_settings.descriptor));
        cudnnErrchk(cudnnSetReduceTensorDescriptor(bias_reduce_settings.descriptor, CUDNN_REDUCE_TENSOR_ADD,
                                                   ::magmadnn::internal::get_cudnn_data_type(static_cast<T>(0)),
                                                   CUDNN_NOT_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES,
                                                   CUDNN_32BIT_INDICES));
        cudnnErrchk(cudnnGetReductionWorkspaceSize(
                          this->get_cudnn_handle(),
                          bias_reduce_settings.descriptor, grad_tmp_descriptor,
                          this->output_tensor->get_cudnn_tensor_descriptor(),
                          &bias_reduce_settings.workspace_size));
        
        cudaErrchk(
            cudaMalloc((void **) &bias_reduce_settings.workspace, bias_reduce_settings.workspace_size * sizeof(T)));

        cudnnErrchk(cudnnDestroyTensorDescriptor(grad_tmp_descriptor));
    }
#endif
}

template class LinearForwardOp<int>;
template class LinearForwardOp<float>;
template class LinearForwardOp<double>;

template <typename T>
LinearForwardOp<T> *linearforward(Operation<T> *input, Operation<T> *weights, bool copy, bool needs_grad) {
    return new LinearForwardOp<T>(input, weights, copy, needs_grad);
}
template LinearForwardOp<int> *linearforward(Operation<int> *input, Operation<int> *weights, bool copy,
                                             bool needs_grad);
template LinearForwardOp<float> *linearforward(Operation<float> *input, Operation<float> *weights, bool copy,
                                               bool needs_grad);
template LinearForwardOp<double> *linearforward(Operation<double> *input, Operation<double> *weights, bool copy,
                                                bool needs_grad);

template <typename T>
LinearForwardOp<T> *linearforward(Operation<T> *input, Operation<T> *weights, Operation<T> *bias, bool copy,
                                  bool needs_grad) {
    return new LinearForwardOp<T>(input, weights, bias, copy, needs_grad);
}
template LinearForwardOp<int> *linearforward(Operation<int> *input, Operation<int> *weights, Operation<int> *bias,
                                             bool copy, bool needs_grad);
template LinearForwardOp<float> *linearforward(Operation<float> *input, Operation<float> *weights,
                                               Operation<float> *bias, bool copy, bool needs_grad);
template LinearForwardOp<double> *linearforward(Operation<double> *input, Operation<double> *weights,
                                                Operation<double> *bias, bool copy, bool needs_grad);

}  // namespace op
}  // namespace magmadnn
