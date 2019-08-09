
#include "compute/crossentropy/crossentropyop.h"

namespace magmadnn {
namespace op {

Operation *crossentropy(Operation *ground_truth, Operation *predicted, bool needs_grad) {
    return negative(reducesum(reducesum(product(ground_truth, log(predicted, true)), 1), 0));
}

}  // namespace op
}  // namespace magmadnn