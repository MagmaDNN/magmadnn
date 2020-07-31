#if defined(MAGMADNN_CMAKE_BUILD)
#include "magmadnn/config.h"
#endif
#include "compute/negative/negative_internal.h"

namespace magmadnn {
namespace internal {

template <typename T>
void negative_full_cpu(Tensor<T> *x, Tensor<T> *out) {
    T *x_ptr = x->get_ptr();
    T *out_ptr = out->get_ptr();
    unsigned int size = out->get_size();
    // TODO Use BLAS
    for (unsigned int i = 0; i < size; i++) {
        out_ptr[i] = -x_ptr[i];
    }
}

template void negative_full_cpu(Tensor<int> *x, Tensor<int> *out);
template void negative_full_cpu(Tensor<float> *x, Tensor<float> *out);
template void negative_full_cpu(Tensor<double> *x, Tensor<double> *out);

template <typename T>
void negative_full(Tensor<T> *x, Tensor<T> *out) {
    if (out->get_memory_type() == HOST) {
        negative_full_cpu(x, out);
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {
        internal::negative_full_device(x, out);
    }
#endif
}
template void negative_full(Tensor<int> *x, Tensor<int> *out);
template void negative_full(Tensor<float> *x, Tensor<float> *out);
template void negative_full(Tensor<double> *x, Tensor<double> *out);

}  // namespace internal
}  // namespace magmadnn
