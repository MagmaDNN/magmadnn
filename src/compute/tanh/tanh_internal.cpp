/**
 * @file sigmoid_internal.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-23
 * 
 * @copyright Copyright (c) 2019
 */
#include "compute/tanh/tanh_internal.h"

namespace skepsi {
namespace internal {

template <typename T>
void tanh_full(tensor<T> *x) {

    if (x->get_memory_type() == HOST) {
        T *x_ptr = x->get_ptr();
        
        for (unsigned int i = 0; i < x->get_size(); i++) {
            x_ptr[i] = tanh(x_ptr[i]);
        }
    }
    #if defined(_HAS_CUDA_)
    else {
        tanh_full_device(x, fast);
    }
    #endif
}
template void tanh_full(tensor<int> *x);
template void tanh_full(tensor<float> *x);
template void tanh_full(tensor<double> *x);

}   // namespace internal
}   // namespace skepsi