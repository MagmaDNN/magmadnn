/**
 * @file inputlayer.cpp
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-26
 * 
 * @copyright Copyright (c) 2019
 */
#include "layer/input/inputlayer.h"

namespace magmadnn {
namespace layer {

template <typename T>
InputLayer<T>::InputLayer(op::Operation<T> *input) : Layer<T>::Layer(input->get_output_shape(), input) {
    init();
}

template <typename T>
std::vector<op::Operation<T> *> InputLayer<T>::get_weights() {
    return {};
}

template <typename T>
void InputLayer<T>::init() {
    this->output = this->input;

    this->name = "InputLayer";
}
template class InputLayer<int>;
template class InputLayer<float>;
template class InputLayer<double>;


template <typename T>
InputLayer<T>* input(op::Operation<T> *input) {
    return new InputLayer<T>(input);
}
template InputLayer<int>* input(op::Operation<int> *input);
template InputLayer<float>* input(op::Operation<float> *input);
template InputLayer<double>* input(op::Operation<double> *input);


}   // layer
}   // magmadnn