
#include "compute/crossentropy/crossentropy_internal.h"

namespace magmadnn {
namespace internal {

template <typename T>
void crossentropy_full(Tensor<T> *x, Tensor<T> *y, Tensor<T> *softmax, Tensor<T> *out) {
    if (out->get_memory_type() == HOST) {
        T *x_ptr = x->get_ptr();
        T *y_ptr = y->get_ptr();
        T *softmax_ptr = softmax->get_ptr();
        T *out_ptr = out->get_ptr();
        unsigned int x_size = x->get_size();
        T x_max = x_ptr[0];
        T exps_sum = (T) 0;
        unsigned int n_rows = x->get_shape(0);

        /* compute max of x */
        for (unsigned int i = 1; i < x_size; i++) {
            if (x_ptr[i] > x_max) {
                x_max = x_ptr[i];
            }
        }

        /* softmax = exp(x- max(x)). also sum exp elements */
        for (unsigned int i = 0; i < x_size; i++) {
            softmax_ptr[i] = exp(x_ptr[i] - x_max);
            exps_sum += softmax_ptr[i];
        }

        /* divide each exp by exps_sum */
        for (unsigned int i = 0; i < x_size; i++) {
            softmax_ptr[i] /= exps_sum;
        }

        for (unsigned int i = 0; i < x_size; i++) {
            out_ptr[0] += y_ptr[i] * log(softmax_ptr[i]);
        }
        out_ptr[0] /= -((T) n_rows);
    }
#if defined(_HAS_CUDA_)
    else {
        crossentropy_full_device(x, y, softmax, out);
    }
#endif
}
template void crossentropy_full(Tensor<int> *x, Tensor<int> *y, Tensor<int> *softmax, Tensor<int> *out);
template void crossentropy_full(Tensor<float> *x, Tensor<float> *y, Tensor<float> *softmax, Tensor<float> *out);
template void crossentropy_full(Tensor<double> *x, Tensor<double> *y, Tensor<double> *softmax, Tensor<double> *out);

}  // namespace internal
}  // namespace magmadnn