#include "compute/log/log_internal.h"

#if defined(MAGMADNN_CMAKE_BUILD)
#include "magmadnn/config.h"
#endif

namespace magmadnn {
namespace internal {

template <typename T>
void log_full_cpu(Tensor<T> *x, Tensor<T> *out, bool stable) {
    T *x_ptr = x->get_ptr();
    T *out_ptr = out->get_ptr();
    unsigned int size = x->get_size();

    T epsilon = (stable) ? static_cast<T>(1E-8) : static_cast<T>(0);

    for (unsigned int i = 0; i < size; i++) {
        out_ptr[i] = log(x_ptr[i] + epsilon);
    }
}
template void log_full_cpu(Tensor<int> *x, Tensor<int> *out, bool stable);
template void log_full_cpu(Tensor<float> *x, Tensor<float> *out, bool stable);
template void log_full_cpu(Tensor<double> *x, Tensor<double> *out, bool stable);

template <typename T>
void log_full(Tensor<T> *x, Tensor<T> *out, bool stable) {
    if (x->get_memory_type() == HOST) {
        log_full_cpu(x, out, stable);
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {
        log_full_device(x, out, stable);
    }
#endif
}
template void log_full(Tensor<int> *x, Tensor<int> *out, bool stable);
template void log_full(Tensor<float> *x, Tensor<float> *out, bool stable);
template void log_full(Tensor<double> *x, Tensor<double> *out, bool stable);

template <typename T>
void log_grad_cpu(Tensor<T> *x, Tensor<T> *grad, Tensor<T> *out, bool stable) {
    T *x_ptr = x->get_ptr();
    T *grad_ptr = grad->get_ptr();
    T *out_ptr = out->get_ptr();
    unsigned int size = out->get_size();

    T epsilon = (stable) ? static_cast<T>(1E-8) : static_cast<T>(0);

    for (unsigned int i = 0; i < size; i++) {
        out_ptr[i] = grad_ptr[i] / (x_ptr[i] + epsilon);
    }
}
template void log_grad_cpu(Tensor<int> *x, Tensor<int> *grad, Tensor<int> *out, bool stable);
template void log_grad_cpu(Tensor<float> *x, Tensor<float> *grad, Tensor<float> *out, bool stable);
template void log_grad_cpu(Tensor<double> *x, Tensor<double> *grad, Tensor<double> *out, bool stable);

template <typename T>
void log_grad(Tensor<T> *x, Tensor<T> *grad, Tensor<T> *out, bool stable) {
    if (x->get_memory_type() == HOST) {
        log_grad_cpu(x, grad, out, stable);
    }
#if defined(MAGMADNN_HAVE_CUDA)
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
