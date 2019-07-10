/**
 * @file outputlayer.h
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-02-26
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
class OutputLayer : public Layer<T> {
   public:
    OutputLayer(op::Operation<T> *input);
    virtual ~OutputLayer();

    virtual std::vector<op::Operation<T> *> get_weights();

   protected:
    void init();
};

template <typename T>
OutputLayer<T> *output(op::Operation<T> *input);

}  // namespace layer
}  // namespace magmadnn