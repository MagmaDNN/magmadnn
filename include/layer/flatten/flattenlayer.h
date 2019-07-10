/**
 * @file flattenlayer.h
 * @author Sedrick Keh
 * @version 1.0
 * @date 2019-07-08
 *
 * @copyright Copyright (c) 2019
 */
#include <vector>
#include "compute/operation.h"
#include "compute/tensor_operations.h"
#include "layer/layer.h"
#include "tensor/tensor.h"

namespace magmadnn {
namespace layer {

template <typename T>
class FlattenLayer : public Layer<T> {
   public:
    FlattenLayer(op::Operation<T> *input);

    virtual std::vector<op::Operation<T> *> get_weights();

   protected:
    void init();
};

/** A new Flatten layer. Typically used after a convolutional layer. Converts the input tensor into a 2d Tensor by
 * flattening it.
 * @tparam T numeric
 * @param input Input to flatten
 * @return FlattenLayer<T>* Flatten layer
 */
template <typename T>
FlattenLayer<T> *flatten(op::Operation<T> *input);

}  // namespace layer
}  // namespace magmadnn