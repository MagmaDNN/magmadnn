/**
 * @file fullyconnectedlayer.h
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

template <typename T>
class FullyConnectedLayer : public Layer<T> {
public:
    FullyConnectedLayer(op::Operation<T> *input, unsigned int hidden_units, bool use_bias=true);
    ~FullyConnectedLayer();

    virtual std::vector<op::Operation<T> *> get_weights();

protected:
    void init();

    unsigned int hidden_units;
    bool use_bias;

    Tensor<T> *weights_tensor;
    Tensor<T> *bias_tensor;

    op::Operation<T> *weights;
    op::Operation<T> *bias;

};

template <typename T>
FullyConnectedLayer<T>* fullyconnected(op::Operation<T> *input, unsigned int hidden_units, bool use_bias=true);

}   // layer
}   // magmadnn