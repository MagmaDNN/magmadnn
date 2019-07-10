/**
 * @file negate.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-06-13
 *
 * @copyright Copyright (c) 2019
 */
#include "math/negate.h"

namespace magmadnn {
namespace math {

template <typename T>
void negate(Tensor<T> *x, Tensor<T> *out) {
    if (out->get_memory_type() == HOST) {
        T *x_ptr = x->get_ptr();
        T *out_ptr = out->get_ptr();
        unsigned int size = out->get_size();

        for (unsigned int i = 0; i < size; i++) {
            out_ptr[i] = -x_ptr[i];
        }
    }
#if defined(_HAS_CUDA_)
    else {
        T alpha = (T) 1;
        T beta = (T) 0;
        cudnnErrchk(cudnnOpTensor(internal::MAGMADNN_SETTINGS->cudnn_handle, CUDNN_OP_TENSOR_NOT, &alpha,
                                  x->get_cudnn_tensor_descriptor(), x->get_ptr(), &beta,
                                  x->get_cudnn_tensor_descriptor(), x->get_ptr(), &beta,
                                  out->get_cudnn_tensor_descriptor(), out->get_ptr()));
    }
#endif
}

}  // namespace math
}  // namespace magmadnn