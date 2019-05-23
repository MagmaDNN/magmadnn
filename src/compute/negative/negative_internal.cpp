
#include "compute/negative/negative_internal.h"

namespace magmadnn {
namespace internal {

template <typename T>
void negative_full(Tensor<T> *x, Tensor<T> *out) {
    if (out->get_memory_type() == HOST) {
        T *x_ptr = x->get_ptr();
        T *out_ptr = out->get_ptr();
        unsigned int size = out->get_size();

        for (unsigned int i = 0; i < size; i++) {
            out_ptr[i] = - x_ptr[i];
        }
    }
    #if defined(_HAS_CUDA_)
    else {
        internal::negative_full_device(x, out);
    }
    #endif
}
template void negative_full(Tensor<int> *x, Tensor<int> *out);
template void negative_full(Tensor<float> *x, Tensor<float> *out);
template void negative_full(Tensor<double> *x, Tensor<double> *out);

}   // namespace op
}   // namespace magmadnn
