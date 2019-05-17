/**
 * @file sigmoid_internal.h
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-23
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once
#include <math.h>
#include "tensor/tensor.h"


namespace magmadnn {
namespace internal {

/** Computes the element-wise sigmoid on x. 
 * @tparam T 
 * @param x pointer to tensor to be sigmoided
 * @param fast if true, then x=1/(1+|x|) is computed instead of normal sigmoid
 */
template <typename T>
void sigmoid_full(Tensor<T> *x, bool fast=true);

#if defined(_HAS_CUDA_)
/** Computes the element-wise sigmoid on a device.
 * @tparam T 
 * @param x tensor with device_ptr
 * @param fast 
 */
template <typename T>
void sigmoid_full_device(Tensor<T> *x, bool fast=true);
#endif

}   // namespace internal
}   // namespace magmadnn