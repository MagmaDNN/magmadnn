/**
 * @file input_layer.h
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-26
 * 
 * @copyright Copyright (c) 2019
 */
#include <vector>
#include "layer.h"
#include "tensor/tensor.h"

namespace skepsi {
namespace layer {

template <typename T>
class input : public layer<T> {
public:
    input(std::vector<unsigned int> input_shape);
    input(tensor<T> *input_tensor);

    void forward();
    void backward();


protected:
    void init();

};

}   // layer
}   // skepsi