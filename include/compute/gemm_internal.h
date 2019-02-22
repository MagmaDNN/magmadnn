/**
 * @file gemm_internal.h
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-22
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once
#include "cblas.h"
#include "magma.h"
#include "tensor/tensor.h"

namespace skepsi {
namespace internal {

template <typename T>
bool gemm_check(tensor<T> *A, tensor<T> *B, tensor<T> *C, unsigned int &M, unsigned int &N, unsigned int &K);

template <typename T>
void gemm_full(T alpha, tensor<T>* A, tensor<T>* B, T beta, tensor<T>* C);


}   // namespace internal
}   // namespace skepsi
