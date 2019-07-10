/**
 * @file dropoutlayer.cpp
 * @author Sedrick Keh
 * @version 1.0
 * @date 2019-02-26
 *
 * @copyright Copyright (c) 2019
 */
#include "layer/dropout/dropoutlayer.h"

namespace magmadnn {
namespace layer {

template <typename T>
DropoutLayer<T>::DropoutLayer(op::Operation<T> *input, float dropout_rate, unsigned long long seed)
    : Layer<T>::Layer(input->get_output_shape(), input), dropout_rate(dropout_rate), seed(seed) {
    init();
}

template <typename T>
DropoutLayer<T>::~DropoutLayer() {}

template <typename T>
std::vector<op::Operation<T> *> DropoutLayer<T>::get_weights() {
    return {};
}

template <typename T>
void DropoutLayer<T>::init() {
    this->output = op::dropout(this->input, dropout_rate, seed);

    this->name = "DropoutLayer";
}
template class DropoutLayer<int>;
template class DropoutLayer<float>;
template class DropoutLayer<double>;

template <typename T>
DropoutLayer<T> *dropout(op::Operation<T> *input, float dropout_rate, unsigned long long seed) {
    return new DropoutLayer<T>(input, dropout_rate, seed);
}
template DropoutLayer<int> *dropout(op::Operation<int> *input, float dropout_rate, unsigned long long seed);
template DropoutLayer<float> *dropout(op::Operation<float> *input, float dropout_rate, unsigned long long seed);
template DropoutLayer<double> *dropout(op::Operation<double> *input, float dropout_rate, unsigned long long seed);

}  // namespace layer
}  // namespace magmadnn