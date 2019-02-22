/**
 * @file geadd_internal.h
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-22
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once
#include "cblas.h"
#include "tensor/tensor.h"

namespace skepsi {
namespace internal {

template <typename T>
bool geadd_check(tensor<T> *A, tensor<T> *B, tensor<T> *C);

template <typename T>
void geadd_full(T alpha, tensor<T> *A, T beta, tensor<T> *B, tensor<T> *C);

#if defined(_HAS_CUDA_)
template <typename T>
void geadd_full_device(unsigned int M, unsigned int N, T alpha, T *A, T beta, T *B, T *C);
#endif

}   // namespace internal
}   // namespace skepsi
