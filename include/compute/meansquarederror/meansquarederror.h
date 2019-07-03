/**
 * @file meansquarederror.h
 * @author Daniel Nichols
 * @version 0.1.0
 * @date 2019-07-03
 * 
 * @copyright Copyright (c) 2019
 * 
 */
#pragma once

#include "compute/operation.h"
#include "compute/add/addop.h"
#include "compute/negative/negativeop.h"
#include "compute/scalarproduct/scalarproductop.h"
#include "compute/pow/powop.h"
#include "compute/reducesum/reducesumop.h"

namespace magmadnn {
namespace op {

template <typename T>
Operation<T> *meansquarederror(Operation<T> *ground_truth, Operation<T> *prediction);

}
}