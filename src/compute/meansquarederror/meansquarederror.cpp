/**
 * @file meansquarederror.cpp
 * @author Daniel Nichols
 * @version 0.1.0
 * @date 2019-07-03
 * 
 * @copyright Copyright (c) 2019
 * 
 */
#include "compute/meansquarederror/meansquarederror.h"

namespace magmadnn {
namespace op {

template <typename T>
Operation<T> *meansquarederror(Operation<T> *ground_truth, Operation<T> *prediction) {
    return op::scalarproduct(1.0f / ground_truth->get_output_shape(0), op::reducesum(op::pow(op::add(ground_truth, op::negative(prediction)), 2)));
}

}
}