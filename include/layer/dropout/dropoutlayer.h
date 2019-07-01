/**
 * @file dropoutlayer.h
 * @author Sedrick Keh
 * @version 0.0.1
 * @date 2019-06-28
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
class DropoutLayer : public Layer<T> {
public:
    DropoutLayer(op::Operation<T> *input, float dropout_rate);
    virtual ~DropoutLayer();

    virtual std::vector<op::Operation<T> *> get_weights();

protected:
    void init();

    float dropout_rate;
};

template <typename T>
DropoutLayer<T>* dropout(op::Operation<T> *input, float dropout_rate);

}   // layer
}   // magmadnn