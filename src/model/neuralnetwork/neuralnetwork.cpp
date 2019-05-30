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
NeuralNetwork<T>::NeuralNetwork(std::vector<layer::Layer<T> *> layers, optimizer::loss_t loss_func, optimizer::Optimizer<T> *optim, nn_params_t params)
: Model<T>::Model(), layers(layers), loss_func(loss_func), optimizer(optim), model_params(params) {

    this->_name = "NeuralNetworkModel";
}

template <typename T>
NeuralNetwork<T>::NeuralNetwork(std::vector<layer::Layer<T> *> layers, optimizer::loss_t loss_func, optimizer::optimizer_t optimizer, nn_params_t params)
: Model<T>::Model(), layers(layers), loss_func(loss_func), model_params(params) {
    this->_name = "NeuralNetworkModel";

    switch (loss_func) {
        case optimizer::CROSS_ENTROPY:
            /* _obj = op::crossentropy(); break; */
            break;
        case optimizer::MSE:
            /* _obj = op::mse(); break; */
            break;
        default:
            std::fprintf(stderr, "Unknown loss func.\n");
    }

    switch (optimizer) {
        case optimizer::SGD:
            this->optimizer = new optimizer::GradientDescent<T> (_obj, this->default_learning_rate); break;
        case optimizer::ADAM:
            std::fprintf(stderr, "ADAM optimizer not yet supported.\n"); break;
        default:
            std::fprintf(stderr, "Unknown optimizer type.\n");
    }
}

template <typename T>
magmadnn_error_t NeuralNetwork<T>::fit(Tensor<T> *x, Tensor<T> *y, metric_t& metric_out, bool verbose) {
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

    /* main training routine */
    for (unsigned int i = 0; i < n_iter; i++) {
        /* copy x into input layer */


        /* forward propagate */


        /* minimize using gradients */

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