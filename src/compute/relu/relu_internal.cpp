/**
 * @file relu_internal.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-01
 * 
 * @copyright Copyright (c) 2019
 */
#include "compute/relu/relu_internal.h"

namespace skepsi {
namespace internal {

template <typename T>
void relu_full(Tensor<T> *x) {
    if (x->get_memory_type() == HOST) {
        T val;
        for (unsigned int i = 0; i < x->get_size(); i++) {
            val = x->get(i);
            if (val < 0) x->set(i, (T) 0);
        }
    }
    #if defined(_HAS_CUDA_)
    else {
        internal::relu_full_device(x);
    }
    #endif
}
template void relu_full(Tensor<int> *x);
template void relu_full(Tensor<float> *x);
template void relu_full(Tensor<double> *x);


}   // namespace internal
}   // namespace skepsi