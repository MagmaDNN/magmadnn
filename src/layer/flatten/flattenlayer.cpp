/**
 * @file flattenlayer.cpp
 * @author Sedrick Keh
 * @version 0.0.1
 * @date 2019-07-08
 * 
 * @copyright Copyright (c) 2019
 */
#include "layer/flatten/flattenlayer.h"

namespace magmadnn {
namespace layer {

template <typename T>
FlattenLayer<T>::FlattenLayer(op::Operation<T> *input) : Layer<T>::Layer(input->get_output_shape(), input) {
    init();
}

template <typename T>
std::vector<op::Operation<T> *> FlattenLayer<T>::get_weights() {
    return {};
}

template <typename T>
void FlattenLayer<T>::init() {
    this->name = "FlattenLayer";

    this->output = op::flatten(this->input);
}
template class FlattenLayer<int>;
template class FlattenLayer<float>;
template class FlattenLayer<double>;


template <typename T>
FlattenLayer<T>* flatten(op::Operation<T> *input) {
    return new FlattenLayer<T>(input);
}
template FlattenLayer<int>* flatten(op::Operation<int> *input);
template FlattenLayer<float>* flatten(op::Operation<float> *input);
template FlattenLayer<double>* flatten(op::Operation<double> *input);


}   // layer
}   // magmadnn