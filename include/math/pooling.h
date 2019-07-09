/**
 * @file pooling.h
 * @author Sedrick Keh
 * @version 1.0
 * @date 2019-07-08
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "tensor/tensor.h"
#include "utilities_internal.h"

enum pooling_mode {
    MAX_POOL,
    AVERAGE_POOL
};

#if defined(_HAS_CUDA_)
#include "cudnn.h"
#endif

namespace magmadnn {
namespace math {

template <typename T>
void pooling(Tensor<T> *x, Tensor<T> *out);

template <typename T>
void pooling_grad(Tensor<T> *x, Tensor<T> *y, Tensor<T> *grad, Tensor<T> *out);


#if defined(_HAS_CUDA_)

struct cudnn_pooling_settings_t {
    cudnnPoolingDescriptor_t poolingDesc;
};

template <typename T>
void pooling_device(Tensor<T> *x, Tensor<T> *out, cudnn_pooling_settings_t settings);

template <typename T>
void pooling_grad_device(Tensor<T> *x, Tensor<T> *y, Tensor<T> *grad, Tensor<T> *out, cudnn_pooling_settings_t settings);

#endif

}
}