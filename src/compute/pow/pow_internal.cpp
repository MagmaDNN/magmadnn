
#include "compute/pow/pow_internal.h"

namespace magmadnn {
namespace internal {

template <typename T>
void pow_grad(Tensor<T> *x, int power, Tensor<T> *grad, Tensor<T> *out) {
    if (out->get_memory_type() == HOST) {
        T *x_ptr = x->get_ptr();
        T *grad_ptr = grad->get_ptr();
        T *out_ptr = out->get_ptr();
        unsigned int size = out->get_size();
        bool grad_is_scalar = T_IS_SCALAR(grad);

        T val;
        unsigned int abs_pow = (power>=0) ? power : -power;
        for (unsigned int i = 0; i < size; i++) {
            /* compute the power */
            val = x_ptr[i];

            for (unsigned int i = 0; i < abs_pow-1; i++) {
                val *= x_ptr[i];
            }

            if (power < 0) val = ((T)1)/val;

            out_ptr[i] = grad_ptr[(grad_is_scalar) ? 0 : i] * ((T)power) * val;
        }
    }
    #if defined(_HAS_CUDA_)
    else {
        internal::pow_grad_device(x, power, grad, out);
    }
    #endif
}
template void pow_grad(Tensor<int> *x, int power, Tensor<int> *input, Tensor<int> *out);
template void pow_grad(Tensor<float> *x, int power, Tensor<float> *input, Tensor<float> *out);
template void pow_grad(Tensor<double> *x, int power, Tensor<double> *input, Tensor<double> *out);

}   // namespace op
}   // namespace magmadnn