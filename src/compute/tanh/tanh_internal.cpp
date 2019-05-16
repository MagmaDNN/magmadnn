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
void tanh_full(Tensor<T> *x) {

    if (x->get_memory_type() == HOST) {
        T *x_ptr = x->get_ptr();
        
        for (unsigned int i = 0; i < x->get_size(); i++) {
            x_ptr[i] = tanh(x_ptr[i]);
        }
    }
    #if defined(_HAS_CUDA_)
    else {
        tanh_full_device(x);
    }
    #endif
}
template void tanh_full(Tensor<int> *x);
template void tanh_full(Tensor<float> *x);
template void tanh_full(Tensor<double> *x);

}   // namespace internal
}   // namespace magmadnn
