#include "compute/pow/powop.h"

#if defined(MAGMADNN_CMAKE_BUILD)
#include "magmadnn/config.h"
#endif
#include "math/pow.h"

namespace magmadnn {
namespace op {

template <typename T>
PowOp<T>::PowOp(Operation<T> *input, int power, bool copy, bool needs_grad)
    : Operation<T>::Operation({input}, needs_grad), input(input), power(power), copy(copy) {
    /* setup code in here */
    this->output_shape = input->get_output_shape();
    this->mem_type = input->get_memory_type();
    this->name = "POW";

    this->output_tensor = new Tensor<T>(this->output_shape, {NONE, {}}, this->mem_type);
}

template <typename T>
Tensor<T> *PowOp<T>::_eval(bool recompute) {
   
   input_tensor = input->eval(recompute);

   if (this->output_tensor->get_memory_type() == HOST) {
      magmadnn::math::pow_cpu(input_tensor, this->power, this->output_tensor);
   }
#if defined(MAGMADNN_HAVE_CUDA)
   else {
      magmadnn::math::pow_device(
            this->get_custream(), input_tensor, this->power,
            this->output_tensor);
      if (!this->get_async()) cudaStreamSynchronize(this->get_custream());
   }
#endif

   return this->output_tensor;
}

template <typename T>
Tensor<T> *PowOp<T>::_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad) {
   /* return gradient in here ... */
   Tensor<T> *out = this->_grad_cache[(uintptr_t) var];
   
   input_tensor = input->eval(false);

   if (out == NULL) {
      out = new Tensor<T>(this->output_shape, {NONE, {}}, this->mem_type);
#if defined(MAGMADNN_HAVE_CUDA)
      out->set_custream(this->get_custream());
      out->set_cublas_handle(this->get_cublas_handle());
#endif
      this->_grad_cache[(uintptr_t) var] = out;
   }

   /* G * power * x^(power-1) */
   // internal::pow_grad(input_tensor, power, grad, out);
   if (out->get_memory_type() == HOST) {
      magmadnn::internal::pow_grad_cpu(input_tensor, power, grad, out);
   }
#if defined(MAGMADNN_HAVE_CUDA)
   else {
      magmadnn::internal::pow_grad_device(
            this->get_custream(), input_tensor, power, grad, out);
      if (!this->get_async()) cudaStreamSynchronize(this->get_custream());
   }
#endif

   return out;
}

template class PowOp<int>;
template class PowOp<float>;
template class PowOp<double>;

template <typename T>
PowOp<T> *pow(Operation<T> *input, int power, bool copy, bool needs_grad) {
    return new PowOp<T>(input, power, copy, needs_grad);
}
template PowOp<int> *pow(Operation<int> *input, int power, bool copy, bool needs_grad);
template PowOp<float> *pow(Operation<float> *input, int power, bool copy, bool needs_grad);
template PowOp<double> *pow(Operation<double> *input, int power, bool copy, bool needs_grad);

}  // namespace op
}  // namespace magmadnn
