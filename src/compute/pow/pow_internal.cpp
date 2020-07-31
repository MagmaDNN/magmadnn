#if defined(MAGMADNN_CMAKE_BUILD)
#include "magmadnn/config.h"
#endif
#include "compute/pow/pow_internal.h"

namespace magmadnn {
namespace internal {

template <typename T>
void pow_grad_cpu(Tensor<T> *x, int power, Tensor<T> *grad, Tensor<T> *out) {
    T *x_ptr = x->get_ptr();
    T *grad_ptr = grad->get_ptr();
    T *out_ptr = out->get_ptr();
    unsigned int size = out->get_size();
    bool grad_is_scalar = T_IS_SCALAR(grad);

    for (unsigned int i = 0; i < size; i++) {
        /* compute the power */
        out_ptr[i] = grad_ptr[(grad_is_scalar) ? 0 : i] * ((T) power) * std::pow((T) x_ptr[i], (T) power - 1);
    }
}

template <>
void pow_grad_cpu(Tensor<int> *x, int power, Tensor<int> *grad, Tensor<int> *out) {
    int *x_ptr = x->get_ptr();
    int *grad_ptr = grad->get_ptr();
    int *out_ptr = out->get_ptr();
    unsigned int size = out->get_size();
    bool grad_is_scalar = T_IS_SCALAR(grad);

    for (unsigned int i = 0; i < size; i++) {
        /* compute the power */
        out_ptr[i] =
            grad_ptr[(grad_is_scalar) ? 0 : i] * power * ((int) std::pow((float) x_ptr[i], (float) (power - 1)));
    }
}
template void pow_grad_cpu(Tensor<float> *x, int power, Tensor<float> *grad, Tensor<float> *out);
template void pow_grad_cpu(Tensor<double> *x, int power, Tensor<double> *grad, Tensor<double> *out);

template <typename T>
void pow_grad(Tensor<T> *x, int power, Tensor<T> *grad, Tensor<T> *out) {
    if (out->get_memory_type() == HOST) {
        pow_grad_cpu(x, power, grad, out);
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {
        internal::pow_grad_device(x, power, grad, out);
    }
#endif
}

// template <>
// void pow_grad(Tensor<int> *x, int power, Tensor<int> *grad, Tensor<int> *out) {
//     if (out->get_memory_type() == HOST) {
//         int *x_ptr = x->get_ptr();
//         int *grad_ptr = grad->get_ptr();
//         int *out_ptr = out->get_ptr();
//         unsigned int size = out->get_size();
//         bool grad_is_scalar = T_IS_SCALAR(grad);

//         for (unsigned int i = 0; i < size; i++) {
//             /* compute the power */
//             out_ptr[i] =
//                 grad_ptr[(grad_is_scalar) ? 0 : i] * power * ((int) std::pow((float) x_ptr[i], (float) (power - 1)));
//         }
//     }
// #if defined(MAGMADNN_HAVE_CUDA)
//     else {
//         internal::pow_grad_device(x, power, grad, out);
//     }
// #endif
// }
template void pow_grad(Tensor<int> *x, int power, Tensor<int> *grad, Tensor<int> *out);
template void pow_grad(Tensor<float> *x, int power, Tensor<float> *grad, Tensor<float> *out);
template void pow_grad(Tensor<double> *x, int power, Tensor<double> *grad, Tensor<double> *out);

}  // namespace internal
}  // namespace magmadnn
