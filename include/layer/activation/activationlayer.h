/**
 * @file activationlayer.h
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-26
 * 
 * @copyright Copyright (c) 2019
 */
#include <vector>
#include "layer/layer.h"
#include "tensor/tensor.h"
#include "compute/operation.h"
#include "compute/tensor_operations.h"

namespace magmadnn {
namespace layer {

enum activation_t {
    SIGMOID,
    TANH,
    RELU
};

template <typename T>
class ActivationLayer : public Layer<T> {
public:
    ActivationLayer(op::Operation<T> *input, activation_t activation_func);
    ~ActivationLayer();

protected:
    void init();

    /* TODO add custom activation function */
    activation_t activation_func;
    
};

template <typename T>
ActivationLayer<T>* activation(op::Operation<T> *input, activation_t activation_func);

}   // layer
}   // magmadnn