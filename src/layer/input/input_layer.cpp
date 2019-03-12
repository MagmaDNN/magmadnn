/**
 * @file input_layer.cpp
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-26
 * 
 * @copyright Copyright (c) 2019
 */
#include "layer/input/input_layer.h"

namespace skepsi {
namespace layer {

template <typename T>
input_layer<T>::input_layer(op::operation<T> *input) : layer<T>::layer(input->get_output_shape(), input) {
    init();
}

template <typename T>
void input_layer<T>::init() {
    this->output = this->input;

    this->name = "InputLayer";
}
template class input_layer<int>;
template class input_layer<float>;
template class input_layer<double>;


template <typename T>
input_layer<T>* input(op::operation<T> *input) {
    return new input_layer<T>(input);
}
template input_layer<int>* input(op::operation<int> *input);
template input_layer<float>* input(op::operation<float> *input);
template input_layer<double>* input(op::operation<double> *input);


}   // layer
}   // skepsi