
#include "compute/transpose/transpose_internal.h"

namespace magmadnn {
namespace internal {

template <typename T>
void transpose_full(Tensor<T> *x, Tensor<T> *out) {
    if (out->get_memory_type() == HOST) {
        unsigned int x_rows = x->get_shape(0);
        unsigned int x_cols = x->get_shape(1);

        for (unsigned int r = 0; r < x_rows; r++) {
            for (unsigned int c = 0; c < x_cols; c++) {
                out->set({(int) c, (int) r}, x->get({(int) r, (int) c}));
            }
        }
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {
        transpose_full_device(x, out);
    }
#endif
}
template void transpose_full(Tensor<int> *x, Tensor<int> *out);
template void transpose_full(Tensor<float> *x, Tensor<float> *out);
template void transpose_full(Tensor<double> *x, Tensor<double> *out);

}  // namespace internal
}  // namespace magmadnn
