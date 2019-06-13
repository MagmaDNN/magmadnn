
#pragma once

#include "tensor/tensor.h"
#include "utilities_internal.h"
#include "cblas.h"
#include "types.h"
#if defined(_HAS_CUDA_)
#include "magma.h"
#include "cudnn_v7.h"
#endif

namespace magmadnn {
namespace internal {

template <typename T>
void tensor_reducesum_full(Tensor<T> *x, unsigned int axis, Tensor<T> *out);


#if defined(_HAS_CUDA_)
template <typename T>
void tensor_reducesum_full_device(Tensor<T> *x, unsigned int axis, Tensor<T> *out);
#endif


/** Col Reduce.
 * @tparam T 
 * @param x 
 * @param out 
 */
template <typename T>
void col_reducesum_full(Tensor<T> *x, Tensor<T> *ones, Tensor<T> *out);


/** Row reduce.
 * @tparam T 
 * @param x 
 * @param out 
 */
template <typename T>
void row_reducesum_full(Tensor<T> *x, Tensor<T> *ones, Tensor<T> *out);


/** Sum of all elements.
 * @tparam T 
 * @param x 
 * @param out 
 */
template <typename T>
void reducesum_full(Tensor<T> *x, Tensor<T> *out);


#if defined(_HAS_CUDA_)
template <typename T>
void reducesum_full_device(Tensor<T> *x, Tensor<T> *out);
#endif


}   // namespace internal
}   // namespace magmadnn