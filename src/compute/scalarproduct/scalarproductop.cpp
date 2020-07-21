/**
 * @file scalarproductop.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-22
 *
 * @copyright Copyright (c) 2019
 */
#include "compute/scalarproduct/scalarproductop.h"

#if defined(MAGMADNN_CMAKE_BUILD)
#include "magmadnn/config.h"
#endif

namespace magmadnn {
namespace op {

template <typename T>
ScalarProductOp<T>::ScalarProductOp(T alpha, Operation<T> *x, bool copy, bool needs_grad)
    : Operation<T>::Operation({x}, needs_grad), alpha(alpha), scalar(NULL), x(x), copy(copy) {
    this->mem_type = x->get_memory_type();
    this->output_shape = x->get_output_shape();

    if (copy) {
        this->output_tensor = new Tensor<T>(x->get_output_shape(), {NONE, {}}, x->get_memory_type());
    }
}

template <typename T>
ScalarProductOp<T>::ScalarProductOp(Operation<T> *scalar, Operation<T> *x, bool copy, bool needs_grad)
    : Operation<T>::Operation({scalar, x}, needs_grad), alpha((T) 1), scalar(scalar), x(x), copy(copy) {
    assert(scalar->get_output_shape().size() == 1 && scalar->get_output_shape()[0] == 1);
    this->mem_type = x->get_memory_type();
    this->output_shape = x->get_output_shape();

    if (copy) {
        this->output_tensor = new Tensor<T>(x->get_output_shape(), {NONE, {}}, x->get_memory_type());
    }
}

template <typename T>
Tensor<T> *ScalarProductOp<T>::_eval(bool recompute) {
    x_tensor = x->eval(recompute);

    if (scalar != NULL) {
        scalar_tensor = scalar->eval(recompute);
        scalar_tensor->get_memory_manager()->sync(true);
        alpha = scalar_tensor->get(0);
    }

    if (!copy) this->output_tensor = x_tensor;

    if (x->get_memory_type() == HOST) {
        magmadnn::internal::scalarproduct_full_cpu(alpha, x_tensor, this->output_tensor);
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {
        magmadnn::internal::scalarproduct_full_device(this->get_custream(), alpha, x_tensor, this->output_tensor);
        if (!this->get_async()) cudaStreamSynchronize(this->get_custream());
    }
#endif

    return this->output_tensor;
}

template <typename T>
Tensor<T> *ScalarProductOp<T>::_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad) {
    if (scalar != NULL) {
        scalar_tensor = scalar->eval(false);
        scalar_tensor->get_memory_manager()->sync(true);

        // internal::scalarproduct_full(scalar_tensor->get(0), grad, grad);
        if (x->get_memory_type() == HOST) {
            magmadnn::internal::scalarproduct_full_cpu(scalar_tensor->get(0), grad, grad);
        }
#if defined(MAGMADNN_HAVE_CUDA)
        else {
            magmadnn::internal::scalarproduct_full_device(this->get_custream(), scalar_tensor->get(0), grad, grad);
            if (!this->get_async()) cudaStreamSynchronize(this->get_custream());
        }
#endif

    } else {
        // internal::scalarproduct_full(alpha, grad, grad);
        if (x->get_memory_type() == HOST) {
            magmadnn::internal::scalarproduct_full_cpu(alpha, grad, grad);
        }
#if defined(MAGMADNN_HAVE_CUDA)
        else {
            magmadnn::internal::scalarproduct_full_device(this->get_custream(), alpha, grad, grad);
            if (!this->get_async()) cudaStreamSynchronize(this->get_custream());
        }
#endif
    }
    return grad;
}

template <typename T>
std::string ScalarProductOp<T>::to_string() {
    if (scalar != NULL) {
        return "( " + scalar->to_string() + " * " + x->to_string() + " )";
    } else {
        return "( __scalar * " + x->to_string() + " )";
    }
}

template class ScalarProductOp<int>;
template class ScalarProductOp<float>;
template class ScalarProductOp<double>;

template <typename T>
ScalarProductOp<T> *scalarproduct(T alpha, Operation<T> *x, bool copy, bool needs_grad) {
    return new ScalarProductOp<T>(alpha, x, copy, needs_grad);
}
template ScalarProductOp<int> *scalarproduct(int alpha, Operation<int> *x, bool copy, bool needs_grad);
template ScalarProductOp<float> *scalarproduct(float alpha, Operation<float> *x, bool copy, bool needs_grad);
template ScalarProductOp<double> *scalarproduct(double alpha, Operation<double> *x, bool copy, bool needs_grad);

template <typename T>
ScalarProductOp<T> *scalarproduct(Operation<T> *scalar, Operation<T> *x, bool copy, bool needs_grad) {
    return new ScalarProductOp<T>(scalar, x, copy, needs_grad);
}
template ScalarProductOp<int> *scalarproduct(Operation<int> *scalar, Operation<int> *x, bool copy, bool needs_grad);
template ScalarProductOp<float> *scalarproduct(Operation<float> *scalar, Operation<float> *x, bool copy,
                                               bool needs_grad);
template ScalarProductOp<double> *scalarproduct(Operation<double> *scalar, Operation<double> *x, bool copy,
                                                bool needs_grad);

}  // namespace op
}  // namespace magmadnn
