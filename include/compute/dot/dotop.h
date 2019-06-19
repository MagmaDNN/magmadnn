/**
 * @file dotop.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-22
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once

#include <cstdio>
#include "compute/operation.h"
#include "compute/matmul/matmulop.h"
#include "compute/product/productop.h"
#include "compute/scalarproduct/scalarproductop.h"

namespace magmadnn {
namespace op {

/** Dot operation. 
 * @tparam T int float double
 * @param a tensor
 * @param b tensor
 * @param copy 
 * @param needs_grad 
 * @return Operation<T>* 
 */
template <typename T>
Operation<T> *dot(Operation<T> *a, Operation<T> *b, bool copy=true, bool needs_grad=true);

}   // namespace op
}   // namespace magmadnn