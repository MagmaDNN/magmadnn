/**
 * @file relu_internal.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-01
 *
 * @copyright Copyright (c) 2019
 */
#include "compute/relu/relu_internal.h"

namespace magmadnn {
namespace internal {

template <typename T>
magmadnn_error_t relu_full(Tensor<T> *x, Tensor<T> *out) {
    if (x->get_memory_type() == HOST) {
        T val;
        for (unsigned int i = 0; i < x->get_size(); i++) {
            val = x->get(i);
            if (val < 0)
                out->set(i, (T) 0);
            else
                out->set(i, val);
        }
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {
        internal::relu_full_device(x, out);
    }
#endif
    return (magmadnn_error_t) 0;
}
template magmadnn_error_t relu_full(Tensor<int> *x, Tensor<int> *out);
template magmadnn_error_t relu_full(Tensor<float> *x, Tensor<float> *out);
template magmadnn_error_t relu_full(Tensor<double> *x, Tensor<double> *out);

}  // namespace internal
}  // namespace magmadnn
