/**
 * @file inputlayer.h
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

namespace skepsi {
namespace layer {

template <typename T>
class InputLayer : public Layer<T> {
public:
    InputLayer(op::Operation<T> *input);

protected:
    void init();

};

template <typename T>
InputLayer<T>* input(op::Operation<T> *input);

}   // layer
}   // skepsi