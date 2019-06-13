/**
 * @file sigmoid_internal.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-23
 * 
 * @copyright Copyright (c) 2019
 */
#include "compute/tanh/tanh_internal.h"

namespace magmadnn {
namespace internal {

template <typename T>
void tanh_full(Tensor<T> *x, Tensor<T> *out) {

    if (x->get_memory_type() == HOST) {
        T *x_ptr = x->get_ptr();
        T *out_ptr = out->get_ptr();
        unsigned int size = out->get_size();
        
        for (unsigned int i = 0; i < size; i++) {
            out_ptr[i] = tanh(x_ptr[i]);
        }
    }
    #if defined(_HAS_CUDA_)
    else {
        tanh_full_device(x, out);
    }
    #endif
}
template void tanh_full(Tensor<int> *x, Tensor<int> *out);
template void tanh_full(Tensor<float> *x, Tensor<float> *out);
template void tanh_full(Tensor<double> *x, Tensor<double> *out);

}   // namespace internal
}   // namespace magmadnn
