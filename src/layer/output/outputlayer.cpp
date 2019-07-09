/**
 * @file outputlayer.cpp
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-02-26
 * 
 * @copyright Copyright (c) 2019
 */
#include "layer/output/outputlayer.h"

namespace magmadnn {
namespace layer {

template <typename T>
OutputLayer<T>::OutputLayer(op::Operation<T> *input) 
    : Layer<T>::Layer(input->get_output_shape(), input) {
    
    init();
}

template <typename T>
std::vector<op::Operation<T> *> OutputLayer<T>::get_weights() {
    return {};
}

template <typename T>
OutputLayer<T>::~OutputLayer() {

}

template <typename T>
void OutputLayer<T>::init() {
    this->name = "OutputLayer";

    this->output = this->input;
}
template class OutputLayer <int>;
template class OutputLayer <float>;
template class OutputLayer <double>;


template <typename T>
OutputLayer<T>* output(op::Operation<T> *input) {
    return new OutputLayer<T> (input);
}
template OutputLayer<int>* output(op::Operation<int> *);
template OutputLayer<float>* output(op::Operation<float> *);
template OutputLayer<double>* output(op::Operation<double> *);


}   // layer
}   // magmadnn