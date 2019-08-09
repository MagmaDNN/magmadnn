
#pragma once

#include "compute/log/logop.h"
#include "compute/negative/negativeop.h"
#include "compute/operation.h"
#include "compute/product/productop.h"
#include "compute/reducesum/reducesumop.h"
#include "math/crossentropy.h"
#include "tensor/tensor.h"
#include "utilities_internal.h"

namespace magmadnn {
namespace op {

/** Returns an operation, which computes the crossentropy between ground_truth and predicted. This must be passed
 * one-hot encoded data. If not one-hot encoded the return values will not be correct or an error may occur. This
 * operation is equivalent to `negative(reducesum(reducesum(product(ground_truth, log(predicted)), axis=1), axis=0))
 * @tparam T int, float, double
 * @param ground_truth the ground_truth (y)
 * @param predicted the predicted values (y_hat)
 * @param copy whether to copy or not copy data
 * @param needs_grad if this operation needs a gradient or not
 * @return Operation<T>* an operation with output size = {1}. This scalar represents the crossentropy.
 */
Operation *crossentropy(Operation *ground_truth, Operation *predicted, bool needs_grad = true);

}  // namespace op
}  // namespace magmadnn