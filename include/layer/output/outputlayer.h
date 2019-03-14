/**
 * @file outputlayer.h
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
class OutputLayer : public Layer<T> {
public:
    OutputLayer(op::Operation<T> *input);
    ~OutputLayer();

protected:
    void init();

};

template <typename T>
OutputLayer<T>* output(op::Operation<T> *input);

}   // layer
}   // skepsi