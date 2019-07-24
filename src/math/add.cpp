/**
 * @file add.cpp
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-06-24
 *
 * @copyright Copyright (c) 2019
 */
#include "math/add.h"

namespace magmadnn {
namespace math {

template <typename T>
void add_in_place(T alpha, const Tensor &x, T beta, Tensor &out) {
    MAGMADNN_ASSERT(T_IS_SAME_DTYPE(x, out), "data type mismatch");
    MAGMADNN_ASSERT(TYPES_MATCH(T, x.dtype()), "data type mismatch");
    MAGMADNN_ASSERT(x.size() == out.size(), "sizes must match");
    // MAGMADNN_ASSERT(T_IS_SAME_MEMORY_TYPE(x, out), "memory type mismatch");

    if (out.get_memory_type() == HOST) {
        const T *x_ptr = x.get_ptr<T>();
        T *out_ptr = out.get_ptr<T>();
        size_t size = out.size();

        for (index_t i = 0; i < size; i++) {
            out_ptr[i] = alpha * x_ptr[i] + beta * out_ptr[i];
        }
    }
#if defined(_HAS_CUDA_)
    else {
        add_in_place_device(alpha, x, beta, out);
    }
#endif
}
#define COMPILE_ADDINPLACE(type) template void add_in_place(type, const Tensor &, type, Tensor &);
CALL_FOR_ALL_TYPES(COMPILE_ADDINPLACE)
#undef COMPILE_ADDINPLACE

#if defined(_HAS_CUDA_)
template <typename T>
void add_in_place_device(T alpha, const Tensor &x, T beta, Tensor &out) {
    cudnnErrchk(cudnnAddTensor(::magmadnn::internal::MAGMADNN_SETTINGS->cudnn_handle, &alpha,
                               x.get_cudnn_tensor_descriptor(), x.get_ptr<T>(), &beta,
                               out.get_cudnn_tensor_descriptor(), out.get_ptr<T>()));
}
#endif

}  // namespace math
}  // namespace magmadnn