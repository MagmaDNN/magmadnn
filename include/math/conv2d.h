/**
 * @file conv2d.h
 * @author Daniel Nichols
 * @version 1.0
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
void conv2d(const Tensor &x, const Tensor &w, Tensor &out);

template <typename T>
void conv2d_grad_data(const Tensor &w, const Tensor &grad, Tensor &out);

template <typename T>
void conv2d_grad_filter(const Tensor &x, const Tensor &grad, Tensor &out);

#if defined(_HAS_CUDA_)

struct conv2d_cudnn_settings {
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnConvolutionFwdAlgo_t algo;
    cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;
    cudnnFilterDescriptor_t filter_desc;
    void *workspace;
    size_t workspace_size;
    void *grad_data_workspace;
    size_t grad_data_workspace_size;
    void *grad_filter_workspace;
    size_t grad_filter_workspace_size;
};

template <typename T>
void conv2d_device(const Tensor &x, const Tensor &w, Tensor &out, conv2d_cudnn_settings settings);

template <typename T>
void conv2d_grad_data_device(const Tensor &w, const Tensor &grad, Tensor &out, conv2d_cudnn_settings settings);

template <typename T>
void conv2d_grad_filter_device(const Tensor &x, const Tensor &grad, Tensor &out, conv2d_cudnn_settings settings);

#endif

}  // namespace math
}  // namespace magmadnn