/**
 * @file residuallayer.cpp
 * @author Sedrick Keh
 * @version 1.0
 * @date 2019-07-11
 *
 * @copyright Copyright (c) 2019
 */
#include "layer/residual/residuallayer.h"

namespace magmadnn {
namespace layer {

template <typename T>
ResidualLayer<T>::ResidualLayer(op::Operation<T> *input,
                                const std::vector<std::pair<unsigned int, unsigned int>> filters,
                                const std::vector<int> out_channels)
    : Layer<T>::Layer(input->get_output_shape(), input), filters(filters), out_channels(out_channels) {
    init();
}

template <typename T>
ResidualLayer<T>::~ResidualLayer() {}

template <typename T>
std::vector<op::Operation<T> *> ResidualLayer<T>::get_weights() {
    return weights;
}

template <typename T>
void ResidualLayer<T>::init() {
    this->name = "Residual";

    assert(filters.size() == out_channels.size());

    /* Input and initial conv layer */
    layers.push_back(layer::input<T>(this->input));
    layers.push_back(layer::conv2d<T>(layers[layers.size() - 1]->out(), {filters[0].first, filters[0].second},
                                      out_channels[0], layer::SAME, {1, 1}, {1, 1}, true, false));

    /* Block of conv + activation layers */
    for (unsigned int i = 1; i < filters.size(); i++) {
        layers.push_back(layer::activation<T>(layers[layers.size() - 1]->out(), layer::RELU));
        layers.push_back(layer::conv2d<T>(layers[layers.size() - 1]->out(), {filters[i].first, filters[i].second},
                                          out_channels[i], layer::SAME, {1, 1}, {1, 1}, true, false));
    }

    /* Shortcut layer */
    layers.push_back(layer::shortcut(layers[layers.size() - 1]->out(), layers[0]->out()));

    /* Final activation layer */
    layers.push_back(layer::activation<T>(layers[layers.size() - 1]->out(), layer::RELU));

    /* Update weights for get_weights() */
    for (unsigned int i = 0; i < layers.size(); i++) {
        std::vector<op::Operation<T> *> layer_i_weights = layers[i]->get_weights();
        for (unsigned int j = 0; j < layer_i_weights.size(); j++) weights.push_back(layer_i_weights[j]);
    }

    this->output = layers[layers.size() - 1]->out();
}

template class ResidualLayer<int>;
template class ResidualLayer<float>;
template class ResidualLayer<double>;

template <typename T>
ResidualLayer<T> *residual(op::Operation<T> *input, const std::vector<std::pair<unsigned int, unsigned int>> filters,
                           const std::vector<int> out_channels) {
    return new ResidualLayer<T>(input, filters, out_channels);
}
template ResidualLayer<int> *residual(op::Operation<int> *, const std::vector<std::pair<unsigned int, unsigned int>>,
                                      const std::vector<int>);
template ResidualLayer<float> *residual(op::Operation<float> *,
                                        const std::vector<std::pair<unsigned int, unsigned int>>,
                                        const std::vector<int>);
template ResidualLayer<double> *residual(op::Operation<double> *,
                                         const std::vector<std::pair<unsigned int, unsigned int>>,
                                         const std::vector<int>);

}  // namespace layer
}  // namespace magmadnn