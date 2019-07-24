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

#include "cblas.h"
#include "tensor/tensor.h"
#include "tensor/tensor_utilities.h"
#include "utilities_internal.h"

#if defined(_HAS_CUDA_)
#include "cudnn.h"
#include "magma.h"

#include "cublas_v2.h"
#endif

namespace magmadnn {
namespace math {

template <typename T>
void reduce_sum(const Tensor &x, int axis, const Tensor &ones, Tensor &out);

#if defined(_HAS_CUDA_)
struct reduce_sum_cudnn_settings_t {
    cudnnReduceTensorDescriptor_t descriptor;
    void *workspace;
    size_t workspace_size;
};

template <typename T>
void reduce_sum_device(const Tensor &x, int axis, Tensor &out, reduce_sum_cudnn_settings_t settings);
#endif

}  // namespace math
}  // namespace magmadnn