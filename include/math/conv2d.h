/**
 * @file conv2d.h
 * @author Daniel Nichols
 * @version 0.1.0
 * @date 2019-06-24
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "tensor/tensor.h"
#include "utilities_internal.h"

#if defined(_HAS_CUDA_)
#include "cudnn.h"
#endif

namespace magmadnn {
namespace math {

template <typename T>
void conv2d(Tensor<T> *x, Tensor<T> *w, Tensor<T> *out);


#if defined(_HAS_CUDA_)

struct conv2d_cudnn_settings {
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnConvolutionFwdAlgo_t algo;
    cudnnFilterDescriptor_t filter_desc;
    void *workspace;
    unsigned int workspace_size;
};

template <typename T>
void conv2d_device(Tensor<T> *x, Tensor<T> *w, Tensor<T> *out, conv2d_cudnn_settings settings);

#endif

}
}