/**
 * @file fullyconnectedlayer.cpp
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-02-26
 *
 * @copyright Copyright (c) 2019
 */
#include "layer/fullyconnected/fullyconnectedlayer.h"

namespace magmadnn {
namespace layer {

template <typename T>
FullyConnectedLayer<T>::FullyConnectedLayer(op::Operation<T>* input, unsigned int hidden_units, bool use_bias)
    : Layer<T>::Layer(input->get_output_shape(), input), hidden_units(hidden_units), use_bias(use_bias) {
    init();
}

template <typename T>
FullyConnectedLayer<T>::~FullyConnectedLayer() {
    delete weights_tensor;
    if (use_bias) delete bias_tensor;
}

template <typename T>
std::vector<op::Operation<T>*> FullyConnectedLayer<T>::get_weights() {
    if (use_bias) {
        return {weights, bias};
    } else {
        return {weights};
    }
}

template <typename T>
unsigned int FullyConnectedLayer<T>::get_num_params() {
    return this->weights->get_output_size() + this->hidden_units;
}

template <typename T>
void FullyConnectedLayer<T>::init() {
    this->name = "FullyConnected";

    /* input is   n_batches x n_classes */

    /* create weight tensor */
    T bound = static_cast<T>(sqrt(2.0 / this->input->get_output_shape(1)));
    this->weights_tensor = new Tensor<T>({this->input->get_output_shape(1), this->hidden_units},
                                         {UNIFORM, {-bound, bound}}, this->input->get_memory_type());
    // this->weights_tensor->fill_memory({UNIFORM, {-bound, bound}});
    this->weights = op::var("__" + this->name + "_layer_weights", this->weights_tensor);

    /* create bias tensor */
    if (use_bias) {
        this->bias_tensor =
            new Tensor<T>({this->input->get_output_shape(0)}, {ZERO, {}}, this->input->get_memory_type());
        this->bias = op::var("__" + this->name + "_layer_bias", this->bias_tensor);
    }

    /*  output = (weights) * (input) + (bias)
        this creates a new tensor and puts it into a new var, which is stored in output. */
    // this->output = op::matmul(this->input, this->weights);

    /* ROW-WISE add bias */
    if (use_bias) {
        this->output = op::linearforward(this->input, this->weights, this->bias);
    } else {
        this->output = op::linearforward(this->input, this->weights);
    }
}
template class FullyConnectedLayer<int>;
template class FullyConnectedLayer<float>;
template class FullyConnectedLayer<double>;

template <typename T>
FullyConnectedLayer<T>* fullyconnected(op::Operation<T>* input, unsigned int hidden_units, bool use_bias) {
    return new FullyConnectedLayer<T>(input, hidden_units, use_bias);
}
template FullyConnectedLayer<int>* fullyconnected(op::Operation<int>*, unsigned int, bool);
template FullyConnectedLayer<float>* fullyconnected(op::Operation<float>*, unsigned int, bool);
template FullyConnectedLayer<double>* fullyconnected(op::Operation<double>*, unsigned int, bool);

}  // namespace layer
}  // namespace magmadnn
