/**
 * @file sigmoid_internal.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-23
 *
 * @copyright Copyright (c) 2019
 */
#include "compute/sigmoid/sigmoid_internal.h"

#include "magmadnn/config.h"

namespace magmadnn {
namespace internal {

template <typename T>
void sigmoid_full_cpu(Tensor<T> *x, Tensor<T> *out, bool fast) {

   T *x_ptr = x->get_ptr();
   T *out_ptr = out->get_ptr();
   unsigned int size = out->get_size();

   if (fast) {
      // fast sigmoid -- fast_sigmoid(x) = x / (1 + |x|)
      for (unsigned int i = 0; i < size; i++) out_ptr[i] = x_ptr[i] / (1 + abs(x_ptr[i]));
   } else {
      // normal sigmoid -- sigmoid(x) = 1 / (1 + exp(-x))
      for (unsigned int i = 0; i < size; i++) out_ptr[i] = 1 / (1 + exp(-x_ptr[i]));
   }

}
template void sigmoid_full_cpu(Tensor<int> *x, Tensor<int> *out, bool fast);
template void sigmoid_full_cpu(Tensor<float> *x, Tensor<float> *out, bool fast);
template void sigmoid_full_cpu(Tensor<double> *x, Tensor<double> *out, bool fast);
   
template <typename T>
void sigmoid_full(Tensor<T> *x, Tensor<T> *out, bool fast) {
    if (x->get_memory_type() == HOST) {
       sigmoid_full_cpu(x, out, fast);
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {
        sigmoid_full_device(x, out, fast);
    }
#endif
}
template void sigmoid_full(Tensor<int> *x, Tensor<int> *out, bool fast);
template void sigmoid_full(Tensor<float> *x, Tensor<float> *out, bool fast);
template void sigmoid_full(Tensor<double> *x, Tensor<double> *out, bool fast);

template <typename T>
void sigmoid_grad_cpu(Tensor<T> *output, Tensor<T> *grad, Tensor<T> *out) {

   T *output_ptr = output->get_ptr();
   T *grad_ptr = grad->get_ptr();
   T *out_ptr = out->get_ptr();
   unsigned int size = out->get_size();

   if (grad->get_size() == 1) {
      for (unsigned int i = 0; i < size; i++) {
         out_ptr[i] = grad_ptr[0] * output_ptr[i] * (((T) 1) - output_ptr[i]);
      }
   } else {
      for (unsigned int i = 0; i < size; i++) {
         out_ptr[i] = grad_ptr[i] * output_ptr[i] * (((T) 1) - output_ptr[i]);
      }
   }
}
template void sigmoid_grad_cpu(Tensor<int> *output, Tensor<int> *grad, Tensor<int> *out);
template void sigmoid_grad_cpu(Tensor<float> *output, Tensor<float> *grad, Tensor<float> *out);
template void sigmoid_grad_cpu(Tensor<double> *output, Tensor<double> *grad, Tensor<double> *out);

template <typename T>
void sigmoid_grad(Tensor<T> *output, Tensor<T> *grad, Tensor<T> *out) {
    /* d s(x) = G * s(x) * (1-s(x)) */

    if (out->get_memory_type() == HOST) {
       sigmoid_grad_cpu(output, grad, out);
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {
        sigmoid_grad_device(output, grad, out);
    }
#endif
}
template void sigmoid_grad(Tensor<int> *output, Tensor<int> *grad, Tensor<int> *out);
template void sigmoid_grad(Tensor<float> *output, Tensor<float> *grad, Tensor<float> *out);
template void sigmoid_grad(Tensor<double> *output, Tensor<double> *grad, Tensor<double> *out);

}  // namespace internal
}  // namespace magmadnn
