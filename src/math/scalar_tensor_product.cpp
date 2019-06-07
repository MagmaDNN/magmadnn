/**
 * @file scalar_tensor_product.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-06-07
 * 
 * @copyright Copyright (c) 2019
 */
#include "math/scalar_tensor_product.h"

namespace magmadnn {
namespace math {

template <typename T>
void scalar_tensor_product(T scalar, Tensor<T> *x, Tensor<T> *out) {
    if (out->get_memory_type()) {
        T *x_ptr = x->get_ptr();
        T *out_ptr = out->get_ptr();
        unsigned int size = out->get_size();

        for (unsigned int i = 0; i < size; i++) {
            out[i] = scalar * x[i];
        }
    }
    #if defined(_HAS_CUDA_)
    else {
        internal::scalar_tensor_product_device(scalar, x, out);
    }
    #endif
}

}
}