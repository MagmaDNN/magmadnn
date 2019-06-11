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
    Tensor<T> *input_tensor, *output_tensor, *predicted, *actual, *output_on_host, *y_on_host;

    /* get the network output from the last layer */
    network_output = this->layers.back()->out();
    output_tensor = network_output->get_output_tensor();

    /* ground truth is given to use by y */
    ground_truth = op::var("y", y);
    
    /* init the argmax tensor */
    predicted = new Tensor<T> ({y->get_shape(0)}, {ZERO,{}}, HOST);
    actual = new Tensor<T> ({y->get_shape(0)}, {ZERO, {}}, HOST);
    output_on_host = new Tensor<T> (output_tensor->get_shape(), {NONE, {}}, HOST);
    y_on_host = new Tensor<T> (y->get_shape(), {NONE,{}}, HOST);

    /* copy y into y_on_host */
    y_on_host->copy_from(*y);
    math::argmax(y_on_host, 0, actual);

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
    double loss = 0.0;
    time_t start_time, end_time;
    unsigned int n_samples = y->get_shape(0);
    unsigned int n_classes = y->get_shape(1);
    unsigned int n_iter = this->model_params.n_epochs;
    Tensor<T> *loss_tensor;

    /* main training routine */
    int n_correct = 0;
    time(&start_time);
    for (unsigned int i = 0; i < n_iter; i++) {
        /* copy x into input layer */
        err = input_tensor->copy_from(*x);

        /* forward pass */
        this->_obj->eval(true);     /* forces evaluation */

        /* minimize using gradients */
        optim->minimize(this->_vars);

        /* calc accuracy */
        output_on_host->copy_from(*output_tensor);
        math::argmax(output_on_host, 0, predicted);
        for (unsigned int i = 0; i < n_samples; i++) {
            if (std::fabs(predicted->get({i}) - actual->get({i})) <= 1E-8) {
                n_correct++;
            }
        }

        /* get the loss from the loss func (_obj) */
        loss_tensor = this->_obj->eval(false);
        loss_tensor->get_memory_manager()->sync();
        loss = loss_tensor->get(0);
    }
    time(&end_time);

    /* update metrics */
    metric_out.accuracy = ((double)n_correct) / (n_samples * n_iter);
    metric_out.loss = loss;
    metric_out.training_time = (end_time - start_time);


    /* free up any memory we used here */
    delete predicted;
    delete actual;
    delete output_on_host;
    delete y_on_host;

    return err;
}

template <typename T>
Tensor<T> *NeuralNetwork<T>::predict(Tensor<T> *sample) {

    Tensor<T> *input_tensor = this->layers.front()->out()->get_output_tensor();

    input_tensor->copy_from(*sample);

    /* Forward propagate -- get the output tensor */
    return this->layers.back()->out()->eval(true);
}

template <typename T>
unsigned int NeuralNetwork<T>::predict_class(Tensor<T> *sample) {

    Tensor<T> *input_tensor = this->layers.front()->out()->get_output_tensor();

    input_tensor->copy_from(*sample);

    this->layers.back()->out()->eval(true);

    Tensor<T> *output_tensor = this->layers.back()->out()->get_output_tensor();
    output_tensor->get_memory_manager()->sync();

    /* TODO -- define argmax in magmadnn::math */
    T val;
    T max = output_tensor->get(0);
    unsigned int arg_max = 0;
    
    for (unsigned int i = 1; i < output_tensor->get_size(); i++) {
        val = output_tensor->get(i);
        if (val > max) {
            max = val;
            arg_max = i;
        }
    }

    return arg_max;
}

template class NeuralNetwork<int>;
template class NeuralNetwork<float>;
template class NeuralNetwork<double>;

}   // namespace model
}   // namespace magmadnn