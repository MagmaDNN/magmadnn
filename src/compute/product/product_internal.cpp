/**
 * @file product_internal.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-21
 * 
 * @copyright Copyright (c) 2019
 */
#include "compute/product/product_internal.h"

namespace magmadnn {
namespace internal {

template <typename T>
void product_full(T alpha, Tensor<T> *a, Tensor<T> *b, Tensor<T> *out) {

    if (a->get_memory_type() == HOST) {
        for (int i = 0; i < (int) a->get_size(); i++) {
            out->set(i, a->get(i) * b->get(i));
        }
    }
    #if defined(_HAS_CUDA_)
    else {
        internal::product_full_device(alpha, a, b, out);
    }
    #endif

}
template void product_full(int alpha, Tensor<int> *a, Tensor<int> *b, Tensor<int> *out);
template void product_full(float alpha, Tensor<float> *a, Tensor<float> *b, Tensor<float> *out);
template void product_full(double alpha, Tensor<double> *a, Tensor<double> *b, Tensor<double> *out);

}   // namespace op
}   // namespace magmadnn