/**
 * @file input_layer.h
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
class input_layer : public layer<T> {
public:
    input_layer(op::operation<T> *input);

protected:
    void init();

};

template <typename T>
input_layer<T>* input(op::operation<T> *input);

}   // layer
}   // skepsi