/**
 * @file sum_internal.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-20
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once

#include <vector>
#include "tensor/tensor.h"

namespace magmadnn {
namespace internal {

/** 
 * @tparam T 
 * @param vals 
 */
template <typename T>
void sum_full(std::vector<Tensor<T> *> &vals, Tensor<T> &out);


#if defined(_HAS_CUDA_)
template <typename T>
void sum_full_device(std::vector<Tensor<T> *> &vals, Tensor<T> &out);
#endif

}   // namespace internal
}   // namespace magmadnn