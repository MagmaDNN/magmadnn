/**
 * @file sigmoid_internal.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-23
 * 
 * @copyright Copyright (c) 2019
 */
#include "compute/sigmoid/sigmoid_internal.h"

namespace skepsi {
namespace internal {

template <typename T>
void sigmoid_full(Tensor<T> *x, bool fast) {

    if (x->get_memory_type() == HOST) {
        T *x_ptr = x->get_ptr();
        
        if (fast) {
            // fast sigmoid -- fast_sigmoid(x) = x / (1 + |x|)
            for(unsigned int i = 0; i < x->get_size(); i++) 
                x_ptr[i] = x_ptr[i] / (1 + abs(x_ptr[i]));
        } else {
            // normal sigmoid -- sigmoid(x) = 1 / (1 + exp(-x))
            for(unsigned int i = 0; i < x->get_size(); i++) 
                x_ptr[i] = 1 / (1 + exp(-x_ptr[i]));
        }
    }
    #if defined(_HAS_CUDA_)
    else {
        sigmoid_full_device(x, fast);
    }
    #endif
}
template void sigmoid_full(Tensor<int> *x, bool fast);
template void sigmoid_full(Tensor<float> *x, bool fast);
template void sigmoid_full(Tensor<double> *x, bool fast);

}   // namespace internal
}   // namespace skepsi