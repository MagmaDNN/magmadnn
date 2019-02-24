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


namespace skepsi {
namespace internal {

template <typename T>
void sigmoid_full(tensor<T> *x, bool fast=true);

#if defined(_HAS_CUDA_)
template <typename T>
void sigmoid_full_device(tensor<T> *x, bool fast=true);
#endif

}   // namespace internal
}   // namespace skepsi