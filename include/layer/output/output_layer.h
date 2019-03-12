/**
 * @file output_layer.h
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
class output_layer : public layer<T> {
public:
    output_layer(op::operation<T> *input);
    ~output_layer();

protected:
    void init();

};

template <typename T>
output_layer<T>* output(op::operation<T> *input);

}   // layer
}   // skepsi