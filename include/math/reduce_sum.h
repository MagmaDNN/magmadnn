/**
 * @file reduce_sum.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-06-14
 *
 * @copyright Copyright (c) 2019
 *
 */
#pragma once

#include "tensor/tensor.h"
#include "magmadnn/utilities_internal.h"

#if defined(MAGMADNN_HAVE_CUDA)
#include "cudnn.h"
#include "magma.h"

#include "cublas_v2.h"
#endif

namespace magmadnn {
namespace math {

template <typename T>
void reduce_sum(Tensor<T> *x, int axis, Tensor<T> *ones, Tensor<T> *out);

#if defined(MAGMADNN_HAVE_CUDA)
struct reduce_sum_cudnn_settings_t {
    cudnnReduceTensorDescriptor_t descriptor;
    void *workspace;
    size_t workspace_size;
    cudnnHandle_t cudnn_handle;
};

template <typename T>
void reduce_sum_device(Tensor<T> *x, int axis, Tensor<T> *out, reduce_sum_cudnn_settings_t settings);
#endif

}  // namespace math
}  // namespace magmadnn
