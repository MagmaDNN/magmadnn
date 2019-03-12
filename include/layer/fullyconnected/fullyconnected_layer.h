/**
 * @file fullyconnected_layer.h
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

template <typename T>
class fullyconnected_layer : public layer<T> {
public:
    fullyconnected_layer(op::operation<T> *input, unsigned int hidden_units, bool use_bias=true);
    ~fullyconnected_layer();

protected:
    void init();

    unsigned int hidden_units;
    bool use_bias;

    tensor<T> *weights_tensor;
    tensor<T> *bias_tensor;

    op::operation<T> *weights;
    op::operation<T> *bias;

};

template <typename T>
fullyconnected_layer<T>* fullyconnected(op::operation<T> *input, unsigned int hidden_units, bool use_bias=true);

}   // layer
}   // skepsi