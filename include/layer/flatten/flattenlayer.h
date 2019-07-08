/**
 * @file flattenlayer.h
 * @author Sedrick Keh
 * @version 0.0.1
 * @date 2019-07-08
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
class FlattenLayer : public Layer<T> {
public:
    FlattenLayer(op::Operation<T> *input);

    virtual std::vector<op::Operation<T> *> get_weights();

protected:
    void init();

};

template <typename T>
FlattenLayer<T>* flatten(op::Operation<T> *input);

}   // layer
}   // magmadnn