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
    op::Operation<T> *network_output, *ground_truth;
    Tensor<T> *output_tensor, *predicted, *actual, *output_on_host, *y_on_host, *x_batch, *y_batch;    

    /* get the network output from the last layer */
    network_output = this->layers.back()->out();
    output_tensor = network_output->get_output_tensor();

    /* construct batches */
    ground_truth = op::var<T>("y", {this->model_params.batch_size, y->get_shape(1)}, {NONE, {}}, output_tensor->get_memory_type());
    y_batch = ground_truth->get_output_tensor();
    
    /* init the argmax tensor */
    predicted = new Tensor<T> ({output_tensor->get_shape(0)}, {ZERO,{}}, HOST); /* this will store the result of the argmax on the output of the network */
    actual = new Tensor<T> ({output_tensor->get_shape(0)}, {ZERO, {}}, HOST);   /* this will store the result of the argmax on the ground_truth */
    output_on_host = new Tensor<T> (output_tensor->get_shape(), {NONE, {}}, HOST);  /* used to move network output onto CPU */
    y_on_host = new Tensor<T> (y_batch->get_shape(), {NONE,{}}, HOST);    /* used to move ground_truth onto CPU */

    /* input tensor is input layer eval */
    /* TODO : this is rather bootleg~ish. there should be an easier way to do this. */
    x_batch = this->layers.front()->out()->get_output_tensor();

    switch (this->loss_func) {
        case optimizer::CROSS_ENTROPY:
            this->_obj = op::crossentropy(ground_truth, network_output); break;
        case optimizer::MSE:
            std::fprintf(stderr, "MSE not yet implemented.\n"); 
            return (magmadnn_error_t) 2;
        default:
            std::fprintf(stderr, "Unknown loss function.\n");
            return (magmadnn_error_t) 2;
    }

    switch (this->optimizer) {
        case optimizer::SGD:
            optim = new optimizer::GradientDescent<T> (this->_obj, this->default_learning_rate / (double)this->model_params.batch_size); break;
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
    unsigned int sample_size = x->get_size() / x->get_shape(0); /* the size of each sample */
    unsigned int ground_truth_sample_size = y->get_size() / y->get_shape(0);
    unsigned int n_iter = this->model_params.n_epochs * (n_samples / this->model_params.batch_size);
    unsigned int cur_sample_idx = 0;
    Tensor<T> *loss_tensor;

    /* main training routine */
    int n_correct = 0;
    time(&start_time);
    for (unsigned int i = 0; i < n_iter; i++) {
        /* load next batch into x*/
        if (cur_sample_idx + this->model_params.batch_size > n_samples) {
            cur_sample_idx = 0;
        }
        err = x_batch->copy_from(*x, cur_sample_idx*sample_size, this->model_params.batch_size * sample_size);
        y_batch->copy_from(*y, cur_sample_idx * ground_truth_sample_size, this->model_params.batch_size * ground_truth_sample_size);

        cur_sample_idx += this->model_params.batch_size;

        /* forward pass */
        this->_obj->eval(true);     /* forces evaluation */

        /* minimize using gradients */
        optim->minimize(this->_vars);

        /* get the argmax of the networks output (on CPU) */
        output_on_host->copy_from(*output_tensor);
        math::argmax(output_on_host, 0, predicted);

        /* get the argmax of the ground truth (on CPU) */
        y_on_host->copy_from(*y_batch);
        math::argmax(y_on_host, 0, actual);

        for (unsigned int j = 0; j < this->model_params.batch_size; j++) {
            if (std::fabs(predicted->get(j) - actual->get(j)) <= 1E-8) {
                n_correct++;
            }
        }

        /* get the loss from the loss func (_obj) */
        loss_tensor = this->_obj->eval(false);
        loss_tensor->get_memory_manager()->sync();
        loss = loss_tensor->get(0);

        if (verbose && i % 10 == 0) {
            printf("Training iteration (%u/%u): accuracy=%.4g loss=%.4g time=%.4g\n",
                i,
                n_iter, 
                n_correct/((double)i*this->model_params.batch_size), 
                loss, 
                (double)time(NULL)-start_time);
        }
    }
    time(&end_time);

    /* update metrics */
    metric_out.accuracy = ((double)n_correct) / ((double) n_samples * n_iter);
    metric_out.loss = loss;
    metric_out.training_time = (double) (end_time - start_time);


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