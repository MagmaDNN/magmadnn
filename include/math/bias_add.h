/**
 * @file bias_add.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-06-23
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "magmadnn/config.h"
#include "tensor/tensor.h"

namespace magmadnn {
namespace math {

template <typename T>
void bias_add_cpu(Tensor<T> *x, Tensor<T> *bias, Tensor<T> *out);
   
template <typename T>
void bias_add(Tensor<T> *x, Tensor<T> *bias, Tensor<T> *out);

#if defined(MAGMADNN_HAVE_CUDA)
template <typename T>
void bias_add_device(Tensor<T> *x, Tensor<T> *bias, Tensor<T> *out);

template <typename T>
void bias_add_device(cudaStream_t custream, Tensor<T> *x, Tensor<T> *bias, Tensor<T> *out)
#endif

}  // namespace math
}  // namespace magmadnn
