
#pragma once

#include "tensor/tensor.h"

namespace magmadnn {
namespace internal {

/** Elementwise division.
 * @tparam T 
 * @param a 
 * @param b
 */
template <typename T>
void tensor_div_tensor_full(Tensor<T> *a, Tensor<T> *b, Tensor<T> *out);


#if defined(_HAS_CUDA_)
template <typename T>
void tensor_div_tensor_full_device(Tensor<T> *a, Tensor<T> *b, Tensor<T> *out);
#endif

/** Tensor-Scalar division.
 * @tparam T 
 * @param a 
 * @param b
 */
template <typename T>
void tensor_div_scalar_full(Tensor<T> *a, T scalar, Tensor<T> *out);


#if defined(_HAS_CUDA_)
template <typename T>
void tensor_div_scalar_full_device(Tensor<T> *a, T scalar, Tensor<T> *out);
#endif


/** Scaler-Tensor division.
 * @tparam T 
 * @param a 
 * @param b
 */
template <typename T>
void scalar_div_tensor_full(T scalar, Tensor<T> *a, Tensor<T> *out);


#if defined(_HAS_CUDA_)
template <typename T>
void scalar_div_tensor_full_device(T scalar, Tensor<T> *a, Tensor<T> *out);
#endif


}   // namespace internal
}   // namespace magmadnn
