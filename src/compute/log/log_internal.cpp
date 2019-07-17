
#include "compute/log/log_internal.h"

namespace magmadnn {
namespace internal {

template <typename T>
void log_full(Tensor<T> *x, Tensor<T> *out, bool stable) {
    if (x->get_memory_type() == HOST) {
        T *x_ptr = x->get_ptr();
        T *out_ptr = out->get_ptr();
        unsigned int size = x->get_size();

        T epsilon = (stable) ? static_cast<T>(1E-8) : static_cast<T>(0);

        for (unsigned int i = 0; i < size; i++) {
            out_ptr[i] = log(x_ptr[i] + epsilon);
        }
    }
#if defined(_HAS_CUDA_)
    else {
        log_full_device(x, out, stable);
    }
#endif
}
template void log_full(Tensor<int> *x, Tensor<int> *out, bool stable);
template void log_full(Tensor<float> *x, Tensor<float> *out, bool stable);
template void log_full(Tensor<double> *x, Tensor<double> *out, bool stable);

template <typename T>
void log_grad(Tensor<T> *x, Tensor<T> *grad, Tensor<T> *out, bool stable) {
    if (x->get_memory_type() == HOST) {
        T *x_ptr = x->get_ptr();
        T *grad_ptr = grad->get_ptr();
        T *out_ptr = out->get_ptr();
        unsigned int size = out->get_size();

        T epsilon = (stable) ? static_cast<T>(1E-8) : static_cast<T>(0);

        for (unsigned int i = 0; i < size; i++) {
            out_ptr[i] = grad_ptr[i] / (x_ptr[i] + epsilon);
        }
    }
#if defined(_HAS_CUDA_)
    else {
        log_grad_device(x, grad, out, stable);
    }
#endif
}
template void log_grad(Tensor<int> *x, Tensor<int> *grad, Tensor<int> *out, bool stable);
template void log_grad(Tensor<float> *x, Tensor<float> *grad, Tensor<float> *out, bool stable);
template void log_grad(Tensor<double> *x, Tensor<double> *grad, Tensor<double> *out, bool stable);

}  // namespace internal
}  // namespace magmadnn
