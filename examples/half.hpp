#pragma once

#include "compute/operation.h"
#include "magma.h"

#define MAGMA_HGEMM_ROWMAJOR(A, B, C, m, n, k, alpha, beta, transf_A, transf_B, lda, ldb, ldc) \
    magma_hgemm(transf_B, transf_A, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc, NULL)

namespace magmadnn {
namespace op {

template <typename T>
class LinearForwardHalfOp : public Operation<T> {
   public:
    LinearForwardHalfOp(Operation<T> *input, Operation<T> *weights, Operation<T> *bias, bool needs_grad = true)
        : Operation<T>::Operation({input, weights, bias}, needs_grad), input(input), weights(weights), bias(bias) {
        this->output_shape = {input->get_output_shape(0), weights->get_output_shape(1)};
        this->mem_type = input->get_memory_type();
        this->name = "LinearForwardHalf";

        this->output_tensor = new Tensor<T>(this->output_shape, {NONE, {}}, this->mem_type);

        magma_malloc((magma_ptr *) input_ptr, input->get_output_size() * sizeof(magmaHalf));
        magma_malloc((magma_ptr *) weights_ptr, weights->get_output_size() * sizeof(magmaHalf));
        magma_malloc((magma_ptr *) out_ptr, this->output_tensor->get_size() * sizeof(magmaHalf));

        /* init bias settings */
        /*cudnnTensorDescriptor_t grad_tmp_descriptor;

        cudnnErrchk( cudnnCreateTensorDescriptor(&grad_tmp_descriptor) );
        cudnnErrchk( cudnnSetTensorDescriptor(grad_tmp_descriptor,
                CUDNN_TENSOR_NCHW,
                ::magmadnn::internal::get_cudnn_data_type((T)0),
                input->get_output_shape(0),
                weights->get_output_shape(1),
                1, 1) );

        cudnnErrchk( cudnnCreateReduceTensorDescriptor
        */
    }

    ~LinearForwardHalfOp() {
        magma_free(input_ptr);
        magma_free(weights_ptr);
        magma_free(out_ptr);
    }

    std::string to_string() { return "LinearForwardHalf"; }

   protected:
    Tensor<T> *_eval(bool recompute) {
        input_tensor = input->eval(recompute);
        weights_tensor = weights->eval(recompute);
        // TODO -- add in bias

        /* convert to half */
        magmablas_convert_sp2hp(input_tensor->get_shape(0), input_tensor->get_shape(1), input_tensor->get_ptr(),
                                input_tensor->get_shape(1), input_ptr, input_tensor->get(1), NULL);

        magmablas_convert_sp2hp(weights_tensor->get_shape(0), weights_tensor->get_shape(1), weights_tensor->get_ptr(),
                                weights_tensor->get_shape(1), weights_ptr, weights_tensor->get_shape(1), NULL);

        /* half-precision matrix multiplication */
        MAGMA_HGEMM_ROWMAJOR(input_ptr, weights_ptr, out_ptr, input_tensor->get_shape(0), input_tensor->get_shape(1),
                             weights_tensor->get_shape(0), (magmaHalf) 1.0f, (magmaHalf) 0.0f, MagmaNoTrans,
                             MagmaNoTrans, input_tensor->get_shape(1), weights_tensor->get_shape(1),
                             this->output_tensor->get_shape(1));

        /* convert back to single */
        /* just convert out back to single */
        // magmablas_convert_hp2sp(input_tensor->get_shape(0), input_tensor->get_shape(1), input_ptr,
        // input_tensor->get_shape(1), input_tensor->get_ptr(), input_tensor->get_shape(1));

        // magmablas_convert_hp2sp(weights_tensor->get_shape(0), weights_tensor->get_shape(1), weights_ptr,
        // input_tensor->get_shape(1), input_tensor->get_ptr(), input_tensor->get_shape(1));

        magmablas_convert_hp2sp(this->output_tensor->get_shape(0), this->output_tensor->get_shape(1), out_ptr,
                                this->output_tensor->get_shape(1), this->output_tensor->get_ptr(),
                                this->output_tensor->get_shape(1), NULL);

        return this->output_tensor;
    }

    Tensor<T> *_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad) {
        Tensor<T> *out = this->_grad_cache[(uintptr_t) var];

        if (var == this->input) {
            this->weights_tensor = this->weights->eval(false);

            if (out == NULL) {
                out =
                    new Tensor<T>({grad->get_shape(0), this->weights_tensor->get_shape(0)}, {NONE, {}}, this->mem_type);
                this->_grad_cache[(uintptr_t) var] = out;
            }
            /* G.W^T */
            // MAGMA_HGEMM_ROWMAJOR(grad->get_weights);
            math::matmul((T) 1, false, grad, true, this->weights_tensor, (T) 0, out);
        } else if (var == this->weights) {
            this->input_tensor = this->input->eval(false);

            if (out == NULL) {
                out = new Tensor<T>({this->input_tensor->get_shape(1), grad->get_shape(1)}, {NONE, {}}, this->mem_type);
                this->_grad_cache[(uintptr_t) var] = out;
            }

            math::matmul((T) 1, true, this->input_tensor, false, grad, (T) 0, out);
        }
        return out;
    }

    Operation<T> *input, *weights, *bias;
    Tensor<T> *input_tensor, *weights_tensor, *bias_tensor;

#if defined(USE_GPU)
    magmaHalf *input_ptr, *weights_ptr, *out_ptr;
    math::reduce_sum_cudnn_settings_t bias_reduce_settings;
#endif
};

template <typename T>
LinearForwardHalfOp<T> *linearforwardhalf(Operation<T> *input, Operation<T> *weights, Operation<T> *bias,
                                          bool needs_grad = true) {
    return new LinearForwardHalfOp<T>(input, weights, bias, needs_grad);
}

}  // namespace op

namespace layer {

template <typename T>
class FullyConnectedHalfLayer : public Layer<T> {
   public:
    FullyConnectedHalfLayer(op::Operation<T> *input, unsigned int hidden_units, bool use_bias = false)
        : Layer<T>::Layer(input->get_output_shape(), input), hidden_units(hidden_units), use_bias(use_bias) {
        init();
    }

    virtual ~FullyConnectedHalfLayer() {
        delete weights_tensor;
        if (use_bias) delete bias_tensor;
    }

    virtual std::vector<op::Operation<T> *> get_weights() {
        if (use_bias) {
            return {weights, bias};
        } else {
            return {weights};
        }
    }

    op::Operation<T> *get_weight() { return weights; }
    op::Operation<T> *get_bias() { return bias; }

   protected:
    void init() {
        this->name = "FullyConnectedHalf";

        T bound = static_cast<T>(sqrt(2.0 / this->input->get_output_shape(1)));
        this->weights_tensor = new Tensor<T>({this->input->get_output_shape(1), this->hidden_units},
                                             {UNIFORM, {-bound, bound}}, this->input->get_memory_type());
        this->weights = op::var("__" + this->name + "_layer_weights", this->weights_tensor);

        if (use_bias) {
            this->bias_tensor =
                new Tensor<T>({this->input->get_output_shape(0)}, {ZERO, {}}, this->input->get_memory_type());
            this->bias = op::var("__" + this->name + "_layer_bias", this->bias_tensor);
        }

        if (use_bias) {
            /* TODO */
        } else {
            this->output = op::linearforwardhalf(this->input, this->weights, bias);
        }
    }

    unsigned int hidden_units;
    bool use_bias;

    Tensor<T> *weights_tensor;
    Tensor<T> *bias_tensor;

    op::Operation<T> *weights;
    op::Operation<T> *bias;
};

template <typename T>
FullyConnectedHalfLayer<T> *fullyconnected_half(op::Operation<T> *input, unsigned int hidden_units,
                                                bool use_bias = false) {
    return new FullyConnectedHalfLayer<T>(input, hidden_units, use_bias);
}

}  // namespace layer
}  // namespace magmadnn