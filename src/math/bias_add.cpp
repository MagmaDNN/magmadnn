/**
 * @file bias_add.cpp
 * @author Daniel Nichols
 * @author Florent Lopez
 * @version 0.1
 * @date 2019-06-23
 *
 * @copyright Copyright (c) 2019
 */
#include "math/bias_add.h"

#include "magmadnn/config.h"

#include <cassert>

namespace magmadnn {
namespace math {

template <typename T>
void bias_add_cpu(Tensor<T> *x, Tensor<T> *bias, Tensor<T> *out) {

   T *x_ptr = x->get_ptr();
   T *bias_ptr = bias->get_ptr();
   T *out_ptr = out->get_ptr();

   unsigned int x_rows = x->get_shape(0);
   unsigned int x_cols = x->get_shape(1);
   // unsigned int x_size = x_rows*x_cols;

   for (unsigned int r = 0; r < x_rows; r++) {
      for (unsigned int c = 0; c < x_cols; c++) {
         out_ptr[r * x_cols + c] = x_ptr[r * x_cols + c] + bias_ptr[r];
      }
   }

}
template void bias_add_cpu(Tensor<int> *x, Tensor<int> *bias, Tensor<int> *out);
template void bias_add_cpu(Tensor<float> *x, Tensor<float> *bias, Tensor<float> *out);
template void bias_add_cpu(Tensor<double> *x, Tensor<double> *bias, Tensor<double> *out);
   
template <typename T>
void bias_add(Tensor<T> *x, Tensor<T> *bias, Tensor<T> *out) {

   assert(T_IS_SAME_MEMORY_TYPE(x, bias) );
   assert(T_IS_SAME_MEMORY_TYPE(bias, out));

   if (out->get_memory_type() == HOST) {
      bias_add_cpu(x, bias, out);
   }
#if defined(MAGMADNN_HAVE_CUDA)
   else {
      bias_add_device(x, bias, out);
   }
#endif
}
template void bias_add(Tensor<int> *x, Tensor<int> *bias, Tensor<int> *out);
template void bias_add(Tensor<float> *x, Tensor<float> *bias, Tensor<float> *out);
template void bias_add(Tensor<double> *x, Tensor<double> *bias, Tensor<double> *out);

#if defined(_USE_CUDNN_BIAS_)
/* temporarily undefined this until cudnn works */

#if defined(MAGMADNN_HAVE_CUDA)
template <typename T>
void bias_add_device(Tensor<T> *x, Tensor<T> *bias, Tensor<T> *out) {
    if (x != out) {
        /* x is not out, so copy x into out since cudnnAddTensor is in-place */
        out->copy_from(*x);
    }

    cudnnDataType_t dat;
    int n, c, h, w, n_s, c_s, h_s, w_s;
    cudnnErrchk(
        cudnnGetTensor4dDescriptor(bias->get_cudnn_tensor_descriptor(), &dat, &n, &c, &h, &w, &n_s, &c_s, &h_s, &w_s));
    printf("bias: %d %d %d %d %d %d %d %d\n", n, c, h, w, n_s, c_s, h_s, w_s);
    cudnnErrchk(
        cudnnGetTensor4dDescriptor(out->get_cudnn_tensor_descriptor(), &dat, &n, &c, &h, &w, &n_s, &c_s, &h_s, &w_s));
    printf("out: %d %d %d %d %d %d %d %d\n", n, c, h, w, n_s, c_s, h_s, w_s);

    T alpha = static_cast<T>(1), beta = static_cast<T>(1);
    cudnnErrchk(cudnnAddTensor(::magmadnn::internal::MAGMADNN_SETTINGS->cudnn_handle, &alpha,
                               bias->get_cudnn_tensor_descriptor(), bias->get_ptr(), &beta,
                               out->get_cudnn_tensor_descriptor(), out->get_ptr()));
}
#endif
#endif

}  // namespace math
}  // namespace magmadnn
