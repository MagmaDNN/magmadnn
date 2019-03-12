/**
 * @file output_layer.cpp
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-26
 * 
 * @copyright Copyright (c) 2019
 */
#include "layer/output/output_layer.h"

namespace skepsi {
namespace layer {

template <typename T>
output_layer<T>::output_layer(op::operation<T> *input) 
    : layer<T>::layer(input->get_output_shape(), input) {
    
    init();
}

template <typename T>
output_layer<T>::~output_layer() {

}

template <typename T>
void output_layer<T>::init() {
    this->name = "OutputLayer";

    this->output = this->input;
}
template class output_layer <int>;
template class output_layer <float>;
template class output_layer <double>;


template <typename T>
output_layer<T>* output(op::operation<T> *input) {
    return new output_layer<T> (input);
}
template output_layer<int>* output(op::operation<int> *);
template output_layer<float>* output(op::operation<float> *);
template output_layer<double>* output(op::operation<double> *);


}   // layer
}   // skepsi