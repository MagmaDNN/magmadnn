/**
 * @file activation_layer.cpp
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-26
 * 
 * @copyright Copyright (c) 2019
 */
#include "layer/activation/activation_layer.h"

namespace skepsi {
namespace layer {

template <typename T>
activation_layer<T>::activation_layer(op::operation<T> *input, activation_t activation_func) 
    : layer<T>::layer(input->get_output_shape(), input), activation_func(activation_func) {
    

    
    init();
}

template <typename T>
activation_layer<T>::~activation_layer() {
    
}

template <typename T>
void activation_layer<T>::init() {
    this->name = "Activation";
    
    switch (this->activation_func) {
        case SIGMOID:
            this->output = op::sigmoid(this->input, false, true); break;
        case TANH:
            this->output = op::tanh(this->input, false); break;
        case RELU:
            fprintf(stderr, "RELU not implemented yet.\n");
        default:
            this->output = op::sigmoid(this->input, false, true); break;
    }
}

template class activation_layer <int>;
template class activation_layer <float>;
template class activation_layer <double>;

template <typename T>
activation_layer<T>* activation(op::operation<T> *input, activation_t activation_func) {
    return new activation_layer<T> (input, activation_func);
}
template activation_layer<int>* activation(op::operation<int>*, activation_t);
template activation_layer<float>* activation(op::operation<float>*, activation_t);
template activation_layer<double>* activation(op::operation<double>*, activation_t);

}   // layer
}   // skepsi