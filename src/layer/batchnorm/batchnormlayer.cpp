/**
 * @file batchnormlayer.cpp
 * @author Sedrick Keh
 * @version 1.0
 * @date 2019-07-24
 *
 * @copyright Copyright (c) 2019
 */
#include "layer/batchnorm/batchnormlayer.h"

namespace magmadnn {
namespace layer {

template <typename T>
BatchNormLayer<T>::BatchNormLayer(op::Operation<T> *input) : Layer<T>::Layer(input->get_output_shape(), input) {
    init();
}

template <typename T>
BatchNormLayer<T>::~BatchNormLayer() {}

template <typename T>
std::vector<op::Operation<T> *> BatchNormLayer<T>::get_weights() {
    return {};
}

template <typename T>
void BatchNormLayer<T>::init() {
    this->output = op::batchnorm(this->input);

    this->name = "BatchNormLayer";
}
template class BatchNormLayer<int>;
template class BatchNormLayer<float>;
template class BatchNormLayer<double>;

template <typename T>
BatchNormLayer<T> *batchnorm(op::Operation<T> *input) {
    return new BatchNormLayer<T>(input);
}
template BatchNormLayer<int> *batchnorm(op::Operation<int> *input);
template BatchNormLayer<float> *batchnorm(op::Operation<float> *input);
template BatchNormLayer<double> *batchnorm(op::Operation<double> *input);

}  // namespace layer
}  // namespace magmadnn