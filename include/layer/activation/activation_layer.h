/**
 * @file activation_layer.h
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

namespace skepsi {
namespace layer {

enum activation_t {
    SIGMOID,
    TANH,
    RELU
};

template <typename T>
class activation_layer : public layer<T> {
public:
    activation_layer(op::operation<T> *input, activation_t activation_func);
    ~activation_layer();

protected:
    void init();

    /* TODO add custom activation function */
    activation_t activation_func;
    
};

template <typename T>
activation_layer<T>* activation(op::operation<T> *input, activation_t activation_func);

}   // layer
}   // skepsi