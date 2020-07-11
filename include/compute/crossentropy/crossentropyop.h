#pragma once

#include "compute/crossentropy/crossentropy_internal.h"
#include "compute/log/logop.h"
#include "compute/negative/negativeop.h"
#include "compute/operation.h"
#include "compute/product/productop.h"
#include "compute/reducesum/reducesumop.h"
#include "compute/scalarproduct/scalarproductop.h"
#include "math/crossentropy.h"
#include "tensor/tensor.h"

namespace magmadnn {
namespace op {

template <typename T>
class CrossEntropyOp : public Operation<T> {
   public:
    CrossEntropyOp(Operation<T> *x, Operation<T> *y, bool copy = true, bool needs_grad = true);
    ~CrossEntropyOp();

    std::string to_string() { return "CrossEntropy(Softmax(" + x->to_string() + "), " + y->to_string() + ")"; }

   protected:
    Tensor<T> *_eval(bool recompute = true);
    Tensor<T> *_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad);

    Operation<T> *x, *y;
    Tensor<T> *x_tensor, *y_tensor, *softmax; /* scratch is used in the interal calc */

    bool copy;
};

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
template <typename T>
Operation<T> *crossentropy(Operation<T> *ground_truth, Operation<T> *predicted, bool copy = true,
                           bool needs_grad = true);

}  // namespace op
}  // namespace magmadnn
