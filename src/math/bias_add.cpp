/**
 * @file bias_add.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-06-23
 *
 * @copyright Copyright (c) 2019
 */
#include "math/bias_add.h"

namespace magmadnn {
namespace math {

template <typename T>
void bias_add(const Tensor &x, const Tensor &bias, Tensor &out) {
    // assert(T_IS_SAME_MEMORY_TYPE(x, bias) && T_IS_SAME_MEMORY_TYPE(bias, out));
    MAGMADNN_ASSERT(TYPES_MATCH(T, x.dtype()) && TYPES_MATCH(T, bias.dtype()) && TYPES_MATCH(T, out.dtype()),
                    "invalid tensor types");

    if (out.get_memory_type() == HOST) {
        const T *x_ptr = x.get_ptr<T>();
        const T *bias_ptr = bias.get_ptr<T>();
        T *out_ptr = out.get_ptr<T>();

        index_t x_rows = x.shape(0);
        index_t x_cols = x.shape(1);
        // unsigned int x_size = x_rows*x_cols;

        /* TODO -- test openmp here */
        for (unsigned int r = 0; r < x_rows; r++) {
            for (unsigned int c = 0; c < x_cols; c++) {
                out_ptr[r * x_cols + c] = x_ptr[r * x_cols + c] + bias_ptr[r];
            }
        }
    }
#if defined(_HAS_CUDA_)
    else {
        bias_add_device<T>(x, bias, out);
    }
#endif
}
#define COMPILE_BIASADD(type) template void bias_add<type>(const Tensor &, const Tensor &, Tensor &out);
CALL_FOR_ALL_TYPES(COMPILE_BIASADD)
#undef COMPILE_BIASADD

#if defined(_USE_CUDNN_BIAS_)
/* temporarily undefined this until cudnn works */

#if defined(_HAS_CUDA_)
template <typename T>
void bias_add_device(const Tensor &x, const Tensor &bias, Tensor &out) {
    cudnnDataType_t dat;
    int n, c, h, w, n_s, c_s, h_s, w_s;
    cudnnErrchk(
        cudnnGetTensor4dDescriptor(bias.get_cudnn_tensor_descriptor(), &dat, &n, &c, &h, &w, &n_s, &c_s, &h_s, &w_s));
    printf("bias: %d %d %d %d %d %d %d %d\n", n, c, h, w, n_s, c_s, h_s, w_s);
    cudnnErrchk(
        cudnnGetTensor4dDescriptor(out.get_cudnn_tensor_descriptor(), &dat, &n, &c, &h, &w, &n_s, &c_s, &h_s, &w_s));
    printf("out: %d %d %d %d %d %d %d %d\n", n, c, h, w, n_s, c_s, h_s, w_s);

    T alpha = static_cast<T>(1), beta = static_cast<T>(1);
    cudnnErrchk(cudnnAddTensor(::magmadnn::internal::MAGMADNN_SETTINGS->cudnn_handle, &alpha,
                               bias.get_cudnn_tensor_descriptor(), bias.get_ptr(), &beta,
                               out.get_cudnn_tensor_descriptor(), out.get_ptr()));
}
#endif
#endif

}  // namespace math
}  // namespace magmadnn