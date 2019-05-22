
#include "compute/scalarproduct/scalarproduct_internal.h"

namespace magmadnn {
namespace internal {

template <typename T>
void scalarproduct_full(T alpha, Tensor<T> *x, Tensor<T> *out) {
    if (x->get_memory_type() == HOST) {

        T *x_ptr, *out_ptr;
        unsigned int size;

        x_ptr = x->get_ptr();
        out_ptr = out->get_ptr();

        size = x->get_size();

        for (unsigned int i = 0; i < size; i++) {
            out_ptr[i] = alpha * x_ptr[i];
        }
    }
    #if defined(_HAS_CUDA_)
    else {
        internal::scalarproduct_device_full(alpha, x, out);
    }
    #endif
}
template void scalarproduct_full(int alpha, Tensor<int> *x, Tensor<int> *out);
template void scalarproduct_full(float alpha, Tensor<float> *x, Tensor<float> *out);
template void scalarproduct_full(double alpha, Tensor<double> *x, Tensor<double> *out);

}   // namespace op
}   // namespace magmadnn
