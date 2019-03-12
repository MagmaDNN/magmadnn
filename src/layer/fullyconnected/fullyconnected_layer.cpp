/**
 * @file fullyconnected_layer.cpp
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-26
 * 
 * @copyright Copyright (c) 2019
 */
#include "layer/fullyconnected/fullyconnected_layer.h"

namespace skepsi {
namespace layer {

template <typename T>
fullyconnected_layer<T>::fullyconnected_layer(op::operation<T> *input, unsigned int hidden_units, bool use_bias) 
    : layer<T>::layer(input->get_output_shape(), input), hidden_units(hidden_units), use_bias(use_bias) {
    
    init();
}

template <typename T>
fullyconnected_layer<T>::~fullyconnected_layer() {
    delete weights_tensor;
    delete bias_tensor;
}

template <typename T>
void fullyconnected_layer<T>::init() {
    this->name = "FullyConnected";

    /* input is   n_batches x n_classes */

    /* create weight tensor */
    this->weights_tensor = new tensor<T> ({this->input->get_output_shape(1), this->hidden_units}, {GLOROT, {(T)0.0, (T)0.5}}, this->input->get_memory_type());
    this->weights = op::var("__"+this->name+"_layer_weights", this->weights_tensor);

    /* create bias tensor */
    this->bias_tensor = new tensor<T> ({1, this->hidden_units}, {GLOROT, {(T)0.0, (T)0.5}}, this->input->get_memory_type());
    this->bias = op::var("__"+this->name+"_layer_bias", this->bias_tensor);

    /*  output = (weights) * (input) + (bias) 
        this creates a new tensor and puts it into a new var, which is stored in output. */
    this->output = op::matmul(this->input, this->weights);
}
template class fullyconnected_layer <int>;
template class fullyconnected_layer <float>;
template class fullyconnected_layer <double>;

template <typename T>
fullyconnected_layer<T>* fullyconnected(op::operation<T> *input, unsigned int hidden_units, bool use_bias) {
    return new fullyconnected_layer<T> (input, hidden_units, use_bias);
}
template fullyconnected_layer<int>* fullyconnected(op::operation<int>*, unsigned int, bool);
template fullyconnected_layer<float>* fullyconnected(op::operation<float>*, unsigned int, bool);
template fullyconnected_layer<double>* fullyconnected(op::operation<double>*, unsigned int, bool);

}   // layer
}   // skepsi