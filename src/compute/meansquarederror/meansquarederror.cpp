/**
 * @file meansquarederror.cpp
 * @author Daniel Nichols
 * @version 1.0
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
    T norm = static_cast<T>(1.0 / ground_truth->get_output_shape(0));
    return op::scalarproduct(norm, op::reducesum(op::pow(op::add(ground_truth, op::negative(prediction)), 2), 0));
}
template Operation<int> *meansquarederror(Operation<int> *ground_truth, Operation<int> *prediction);
template Operation<float> *meansquarederror(Operation<float> *ground_truth, Operation<float> *prediction);
template Operation<double> *meansquarederror(Operation<double> *ground_truth, Operation<double> *prediction);

}  // namespace op
}  // namespace magmadnn