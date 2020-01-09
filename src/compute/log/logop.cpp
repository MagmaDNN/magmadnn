#include "compute/log/logop.h"

#include "magmadnn/config.h"

namespace magmadnn {
namespace op {

template <typename T>
LogOp<T>::LogOp(Operation<T> *x, bool stable, bool copy, bool needs_grad)
    : Operation<T>::Operation({x}, needs_grad), x(x), stable(stable), copy(copy) {
    this->output_shape = x->get_output_shape();
    this->mem_type = x->get_memory_type();

    if (copy) {
        this->output_tensor = new Tensor<T>(this->output_shape, {NONE, {}}, this->mem_type);
    } else {
        std::fprintf(stderr, "no-copy logop not supported.\n");
    }
}

template <typename T>
Tensor<T> *LogOp<T>::_eval(bool recompute) {
    x_tensor = x->eval(recompute);

    if (!copy) this->output_tensor = x_tensor;

    if (this->output_tensor->get_memory_type() == HOST) {
       magmadnn::internal::log_full_cpu(x_tensor, this->output_tensor, stable);
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {
       magmadnn::internal::log_full_device(
             this->get_custream(), x_tensor, this->output_tensor, stable);
       if (!this->get_async()) cudaStreamSynchronize(this->get_custream());
    }
#endif

    return this->output_tensor;
}

template <typename T>
Tensor<T> *LogOp<T>::_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad) {
    /* TODO : grad * (1/x) */
    Tensor<T> *out;

    this->x_tensor = x->eval(false); /* don't recompute x if we don't have to */

    out = this->_grad_cache[(uintptr_t) var];
    if (out == NULL) {
        out = new Tensor<T>(this->output_shape, {NONE, {}}, this->mem_type);
#if defined(MAGMADNN_HAVE_CUDA)
        out->set_custream(this->get_custream());
        out->set_cublas_handle(this->get_cublas_handle());
#endif
        this->_grad_cache[(uintptr_t) var] = out;
    }

    if (out->get_memory_type() == HOST) {
       magmadnn::internal::log_grad_cpu(x_tensor, grad, out, stable);
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {
       magmadnn::internal::log_grad_device(
             this->get_custream(), x_tensor, grad, out, stable);
       if (!this->get_async()) cudaStreamSynchronize(this->get_custream());
    }
#endif

    return out;
}

template class LogOp<int>;
template class LogOp<float>;
template class LogOp<double>;

template <typename T>
LogOp<T> *log(Operation<T> *x, bool stable, bool copy, bool needs_grad) {
    return new LogOp<T>(x, stable, copy, needs_grad);
}
template LogOp<int> *log(Operation<int> *x, bool stable, bool copy, bool needs_grad);
template LogOp<float> *log(Operation<float> *x, bool stable, bool copy, bool needs_grad);
template LogOp<double> *log(Operation<double> *x, bool stable, bool copy, bool needs_grad);

}  // namespace op
}  // namespace magmadnn
