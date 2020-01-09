#include "magmadnn/config.h"
#include "compute/negative/negative_internal.h"
#include "compute/negative/negativeop.h"

namespace magmadnn {
namespace op {

template <typename T>
NegativeOp<T>::NegativeOp(Operation<T> *x, bool copy, bool needs_grad)
    : Operation<T>::Operation({x}, needs_grad), x(x), copy(copy) {
    this->output_shape = x->get_output_shape();
    this->mem_type = x->get_memory_type();

    if (copy) {
        this->output_tensor = new Tensor<T>(this->output_shape, {NONE, {}}, this->mem_type);
    }

    this->_grad_cache[(uintptr_t) x] = NULL;
}

template <typename T>
Tensor<T> *NegativeOp<T>::_eval(bool recompute) {
   
   x_tensor = x->eval(recompute);

   if (!copy) this->output_tensor = x_tensor;

   // internal::negative_full(x_tensor, this->output_tensor);
   if (this->output_tensor->get_memory_type() == HOST) {
      magmadnn::internal::negative_full_cpu(x_tensor, this->output_tensor);
   }
#if defined(MAGMADNN_HAVE_CUDA)
   else {
      // magmadnn::internal::negative_full_device(x_tensor, this->output_tensor);
      magmadnn::internal::negative_full_device(
            this->get_custream(), x_tensor, this->output_tensor);
      if (!this->get_async()) cudaStreamSynchronize(this->get_custream());
   }
#endif
    
   return this->output_tensor;
}

template <typename T>
Tensor<T> *NegativeOp<T>::_grad(
      Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad) {

   /* grad : -grad */
   if (grad->get_memory_type() == HOST) {
      magmadnn::internal::negative_full_cpu(grad, grad);
   }
#if defined(MAGMADNN_HAVE_CUDA)
   else {
      magmadnn::internal::negative_full_device(
            this->get_custream(), grad, grad);
      if (!this->get_async()) cudaStreamSynchronize(this->get_custream());
   }
#endif

   return grad;
}

template class NegativeOp<int>;
template class NegativeOp<float>;
template class NegativeOp<double>;

template <typename T>
NegativeOp<T> *negative(Operation<T> *x, bool copy, bool needs_grad) {
    return new NegativeOp<T>(x, copy, needs_grad);
}
template NegativeOp<int> *negative(Operation<int> *x, bool copy, bool needs_grad);
template NegativeOp<float> *negative(Operation<float> *x, bool copy, bool needs_grad);
template NegativeOp<double> *negative(Operation<double> *x, bool copy, bool needs_grad);

}  // namespace op
}  // namespace magmadnn
