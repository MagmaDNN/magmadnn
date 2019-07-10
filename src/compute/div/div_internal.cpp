
#include "compute/div/div_internal.h"

namespace magmadnn {
namespace internal {

template <typename T>
void tensor_div_tensor_full(Tensor<T> *a, Tensor<T> *b, Tensor<T> *out) {
    if (out->get_memory_type() == HOST) {
        T *a_ptr = a->get_ptr();
        T *b_ptr = b->get_ptr();
        T *out_ptr = out->get_ptr();
        unsigned int size = out->get_size();

        for (unsigned int i = 0; i < size; i++) {
            if (b_ptr[i] == (T) 0) assert(false);
            out_ptr[i] = a_ptr[i] / b_ptr[i];
        }
    }
#if defined(_HAS_CUDA_)
    else {
        tensor_div_tensor_full_device(a, b, out);
    }
#endif
}
template void tensor_div_tensor_full(Tensor<int> *a, Tensor<int> *b, Tensor<int> *out);
template void tensor_div_tensor_full(Tensor<float> *a, Tensor<float> *b, Tensor<float> *out);
template void tensor_div_tensor_full(Tensor<double> *a, Tensor<double> *b, Tensor<double> *out);

template <typename T>
void tensor_div_scalar_full(Tensor<T> *a, T scalar, Tensor<T> *out) {
    if (scalar == (T) 0) assert(false);

    if (out->get_memory_type() == HOST) {
        T *a_ptr = a->get_ptr();
        T *out_ptr = out->get_ptr();
        unsigned int size = out->get_size();

        for (unsigned int i = 0; i < size; i++) {
            out_ptr[i] = a_ptr[i] / scalar;
        }
    }
#if defined(_HAS_CUDA_)
    else {
        tensor_div_scalar_full_device(a, scalar, out);
    }
#endif
}
template void tensor_div_scalar_full(Tensor<int> *a, int scalar, Tensor<int> *out);
template void tensor_div_scalar_full(Tensor<float> *a, float scalar, Tensor<float> *out);
template void tensor_div_scalar_full(Tensor<double> *a, double scalar, Tensor<double> *out);

template <typename T>
void scalar_div_tensor_full(T scalar, Tensor<T> *a, Tensor<T> *out) {
    if (out->get_memory_type() == HOST) {
        T *a_ptr = a->get_ptr();
        T *out_ptr = out->get_ptr();
        unsigned int size = out->get_size();

        for (unsigned int i = 0; i < size; i++) {
            if (a_ptr[i] == (T) 0) assert(false);
            out_ptr[i] = scalar / a_ptr[i];
        }
    }
#if defined(_HAS_CUDA_)
    else {
        scalar_div_tensor_full_device(scalar, a, out);
    }
#endif
}
template void scalar_div_tensor_full(int scalar, Tensor<int> *b, Tensor<int> *out);
template void scalar_div_tensor_full(float scalar, Tensor<float> *b, Tensor<float> *out);
template void scalar_div_tensor_full(double scalar, Tensor<double> *b, Tensor<double> *out);

}  // namespace internal
}  // namespace magmadnn
