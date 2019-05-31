/**
 * @file neuralnetwork.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-30
 * 
 * @copyright Copyright (c) 2019
 */
#include "model/neuralnetwork/neuralnetwork.h"

namespace magmadnn {
namespace model {

template <typename T>
NeuralNetwork<T>::NeuralNetwork(std::vector<layer::Layer<T> *> layers, optimizer::loss_t loss_func, optimizer::optimizer_t optimizer, nn_params_t params)
: Model<T>::Model(), layers(layers), loss_func(loss_func), optimizer(optimizer), model_params(params) {
    this->_name = "NeuralNetworkModel";

    typename std::vector<layer::Layer<T> *>::iterator vit;
    typename std::vector<op::Operation<T> *>::iterator it;
    std::vector<op::Operation<T> *> tmp_vars;

    for (vit = layers.begin(); vit != layers.end(); vit++) {
        tmp_vars = (*vit)->get_weights();
        this->_vars.insert(this->_vars.end(), tmp_vars.begin(), tmp_vars.end());
    }

}

template <typename T>
magmadnn_error_t NeuralNetwork<T>::fit(Tensor<T> *x, Tensor<T> *y, metric_t& metric_out, bool verbose) {
    /* init */
    optimizer::Optimizer<T> *optim;
    op::Operation<T> *network_output;
    op::Operation<T> *ground_truth;
    Tensor<T> *input_tensor;

    /* get the network output from the last layer */
    network_output = this->layers.back()->out();

    /* ground truth is given to use by y */
    ground_truth = op::var("y", y);

    /* input tensor is input layer eval */
    /* TODO : this is rather bootleg~ish. there should be an easier way to do this. */
    input_tensor = this->layers.front()->out()->eval();

    switch (this->loss_func) {
        case optimizer::CROSS_ENTROPY:
            this->_obj = op::crossentropy(network_output, ground_truth); break;
        case optimizer::MSE:
            std::fprintf(stderr, "MSE not yet implemented.\n"); break;
        default:
            std::fprintf(stderr, "Unknown loss function.\n");
    }

    switch (this->optimizer) {
        case optimizer::SGD:
            optim = new optimizer::GradientDescent<T> (this->_obj, this->default_learning_rate); break;
        case optimizer::ADAM:
            std::fprintf(stderr, "Adam not yet implemented.\n");
            return (magmadnn_error_t) 2;
        default:
            std::fprintf(stderr, "Unknown optimizer!\n");
            return (magmadnn_error_t) 2;
    }

    /* Neural Network training Routine.
        1. Copy x tensor into input layer
        2. Forward propagate layer
        3. Call minimize on the optimizer
        4. Go back to 1
    */

    magmadnn_error_t err = (magmadnn_error_t) 0;
    double accuracy = 1.0;
    double loss = 0.0;
    double training_time = 0.0;
    unsigned int n_iter = this->model_params.n_epochs;
    Tensor<T> *loss_tensor;

    /* main training routine */
    for (unsigned int i = 0; i < n_iter; i++) {
        /* copy x into input layer */
        err = input_tensor->copy_from(*x);

        /* minimize using gradients */
        optim->minimize(this->_vars);

        /* get the loss from the loss func (_obj) */
        loss_tensor = this->_obj->eval(false);
        loss_tensor->get_memory_manager()->sync();
        loss = loss_tensor->get(0);
    }

    /* update metrics */
    metric_out.accuracy = accuracy;
    metric_out.loss = loss;
    metric_out.training_time = training_time;

    return err;
}

template <typename T>
Tensor<T> *NeuralNetwork<T>::predict(Tensor<T> *sample) {
    return NULL;
}

template <typename T>
unsigned int NeuralNetwork<T>::predict_class(Tensor<T> *sample) {

    return 0;
}

template class NeuralNetwork<int>;
template class NeuralNetwork<float>;
template class NeuralNetwork<double>;

}   // namespace model
}   // namespace magmadnn