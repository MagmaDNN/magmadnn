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

}   // namespace internal
}   // namespace skepsi