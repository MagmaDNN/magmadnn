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

    if (out->get_memory_type() == HOST) {
        T *a_ptr = a->get_ptr();
        T *b_ptr = b->get_ptr();
        T *out_ptr = out->get_ptr();
        unsigned int size = out->get_size();
        for (unsigned int i = 0; i < size; i++) {
            out_ptr[i] = alpha * a_ptr[i] * b_ptr[i];
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


template <typename T>
void scalar_tensor_product_full(T scalar, Tensor<T> *a, Tensor<T> *out) {

    if (out->get_memory_type() == HOST) {
        T *a_ptr = a->get_ptr();
        T *out_ptr = out->get_ptr();
        unsigned int size = out->get_size();
        for (unsigned int i = 0; i < size; i++) {
            out_ptr[i] = scalar * a_ptr[i];
        }
    }
    #if defined(_HAS_CUDA_)
    else {
        internal::scalar_tensor_product_full_device(scalar, a, out);
    }
    #endif

}
template void scalar_tensor_product_full(int scalar, Tensor<int> *a, Tensor<int> *out);
template void scalar_tensor_product_full(float scalar, Tensor<float> *a, Tensor<float> *out);
template void scalar_tensor_product_full(double scalar, Tensor<double> *a, Tensor<double> *out);

}   // namespace op
}   // namespace magmadnn