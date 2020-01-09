/**
 * @file pooling.cpp
 * @author Sedrick Keh
 * @version 1.0
 * @date 2019-07-08
 *
 * @copyright Copyright (c) 2019
 */
#include "math/pooling.h"

#include <cassert>

#include "magmadnn/config.h"

namespace magmadnn {
namespace math {

template <typename T>
void pooling(Tensor<T> *x, Tensor<T> *out) {
    assert(T_IS_SAME_MEMORY_TYPE(x, out));

    if (out->get_memory_type() == HOST) {
        fprintf(stderr, "__pooling CPU not supported yet.\n");
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {
        fprintf(stderr, "For Pooling GPU please use pooling_device.\n");
    }
#endif
}
template void pooling(Tensor<int> *x, Tensor<int> *out);
template void pooling(Tensor<float> *x, Tensor<float> *out);
template void pooling(Tensor<double> *x, Tensor<double> *out);

template <typename T>
void pooling_grad(Tensor<T> *x, Tensor<T> *y, Tensor<T> *grad, Tensor<T> *out) {

   assert(T_IS_SAME_MEMORY_TYPE(x, y));
   assert(T_IS_SAME_MEMORY_TYPE(y, grad));
   assert(T_IS_SAME_MEMORY_TYPE(grad, out));

    if (out->get_memory_type() == HOST) {
        fprintf(stderr, "__Pooling_grad CPU not supported yet.\n");
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {
        fprintf(stderr, "For pooling_grad GPU please use pooling_grad_device.\n");
    }
#endif
}
template void pooling_grad(Tensor<int> *x, Tensor<int> *y, Tensor<int> *grad, Tensor<int> *out);
template void pooling_grad(Tensor<float> *x, Tensor<float> *y, Tensor<float> *grad, Tensor<float> *out);
template void pooling_grad(Tensor<double> *x, Tensor<double> *y, Tensor<double> *grad, Tensor<double> *out);

#if defined(MAGMADNN_HAVE_CUDA)

template <typename T>
void pooling_device(Tensor<T> *x, Tensor<T> *out, cudnn_pooling_settings_t settings) {

   T alpha = static_cast<T>(1), beta = static_cast<T>(0);

   cudnnErrchk(
         cudnnPoolingForward(
               settings.handle, settings.poolingDesc, &alpha,
               x->get_cudnn_tensor_descriptor(), x->get_ptr(), &beta,
               out->get_cudnn_tensor_descriptor(), out->get_ptr()));
}
template void pooling_device(Tensor<int> *x, Tensor<int> *out, cudnn_pooling_settings_t settings);
template void pooling_device(Tensor<float> *x, Tensor<float> *out, cudnn_pooling_settings_t settings);
template void pooling_device(Tensor<double> *x, Tensor<double> *out, cudnn_pooling_settings_t settings);

template <typename T>
void pooling_grad_device(
      Tensor<T> *x, Tensor<T> *y, Tensor<T> *grad, Tensor<T> *out,
      cudnn_pooling_settings_t settings) {
   
    T alpha = static_cast<T>(1), beta = static_cast<T>(0);

    cudnnErrchk(
          cudnnPoolingBackward(
                settings.handle, settings.poolingDesc, &alpha,
                y->get_cudnn_tensor_descriptor(), y->get_ptr(),
                grad->get_cudnn_tensor_descriptor(), grad->get_ptr(),
                x->get_cudnn_tensor_descriptor(), x->get_ptr(), &beta,
                out->get_cudnn_tensor_descriptor(), out->get_ptr()));
}

template void pooling_grad_device(
      Tensor<int> *x, Tensor<int> *y, Tensor<int> *grad, Tensor<int> *out,
      cudnn_pooling_settings_t settings);
template void pooling_grad_device(
      Tensor<float> *x, Tensor<float> *y, Tensor<float> *grad, Tensor<float> *out,
      cudnn_pooling_settings_t settings);
template void pooling_grad_device(
      Tensor<double> *x, Tensor<double> *y, Tensor<double> *grad, Tensor<double> *out,
      cudnn_pooling_settings_t settings);

#endif

}  // namespace math
}  // namespace magmadnn
