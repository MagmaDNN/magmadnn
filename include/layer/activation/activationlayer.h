/**
 * @file activationlayer.h
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

enum activation_t { SIGMOID, TANH, RELU, SOFTMAX };

template <typename T>
class ActivationLayer : public Layer<T> {
   public:
    ActivationLayer(op::Operation<T> *input, activation_t activation_func);
    ~ActivationLayer();

    virtual std::vector<op::Operation<T> *> get_weights();

   protected:
    void init();

    /* TODO add custom activation function */
    activation_t activation_func;
};

template <typename T>
ActivationLayer<T> *activation(op::Operation<T> *input, activation_t activation_func);

}  // namespace layer
}  // namespace magmadnn