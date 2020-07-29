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
NeuralNetwork<T>::NeuralNetwork(std::vector<layer::Layer<T> *> layers, optimizer::loss_t loss_func,
                                optimizer::optimizer_t optimizer, nn_params_t params)
    : Model<T>::Model(), layers(layers), loss_func(loss_func), optimizer(optimizer), model_params(params) {
    this->_name = "NeuralNetworkModel";

    typename std::vector<layer::Layer<T> *>::iterator vit;
    typename std::vector<op::Operation<T> *>::iterator it;
    std::vector<op::Operation<T> *> tmp_vars;

    /* copy the weights from each layer into _vars */
    for (vit = layers.begin(); vit != layers.end(); vit++) {
        tmp_vars = (*vit)->get_weights();
        this->_vars.insert(this->_vars.end(), tmp_vars.begin(), tmp_vars.end());
    }

    /* get pointers to network back and front operations */
    this->network_input_op_ptr = layers.front()->out();
    this->network_output_op_ptr = layers.back()->out();

    /* get pointers to network back and front tensors */
    this->network_input_tensor_ptr = this->network_input_op_ptr->get_output_tensor();
    this->network_output_tensor_ptr = this->network_output_op_ptr->get_output_tensor();

    /* init ground truth pointers */
    this->ground_truth_op_ptr =
        op::var<T>(this->_name + "::ground_truth", {params.batch_size, network_output_tensor_ptr->get_shape().back()},
                   {NONE, {}}, network_output_tensor_ptr->get_memory_type());
    this->ground_truth_tensor_ptr = this->ground_truth_op_ptr->get_output_tensor();

    /* init loss function -- _obj */
    switch (loss_func) {
        case optimizer::CROSS_ENTROPY:
            this->_obj = op::crossentropy(this->ground_truth_op_ptr, this->network_output_op_ptr);
            break;
        case optimizer::MSE:
            this->_obj = op::meansquarederror(this->ground_truth_op_ptr, this->network_output_op_ptr);
            break;
        default:
            std::fprintf(stderr, "Unknown loss function.\n");
            break;
    }
    this->_obj_tensor_ptr = this->_obj->get_output_tensor();

    /* init optimizer */
    switch (optimizer) {
        case optimizer::SGD:
            this->optim = new optimizer::GradientDescent<T>(
                static_cast<T>(params.learning_rate),
                static_cast<T>(params.momentum));
            break;
        case optimizer::ADAGRAD:
            this->optim =
                new optimizer::AdaGrad<T>(static_cast<T>(params.learning_rate));
            break;
        case optimizer::RMSPROP:
            this->optim =
                new optimizer::RMSProp<T>(static_cast<T>(params.learning_rate),
                                          static_cast<T>(params.decaying_factor));
            break;
        case optimizer::ADAM:
            this->optim =
                new optimizer::Adam<T>(static_cast<T>(params.learning_rate),
                                       static_cast<T>(params.beta1), static_cast<T>(params.beta2));
            break;
        default:
            std::fprintf(stderr, "Unknown optimizer.\n");
            break;
    }
}

template <typename T>
NeuralNetwork<T>::NeuralNetwork(std::vector<layer::Layer<T> *> layers, optimizer::loss_t loss_func,
                                optimizer::Optimizer<T> *optim, nn_params_t params)
    : Model<T>::Model(), layers(layers), loss_func(loss_func), model_params(params), optim(optim) {
    this->_name = "NeuralNetworkModel";

    typename std::vector<layer::Layer<T> *>::iterator vit;
    typename std::vector<op::Operation<T> *>::iterator it;
    std::vector<op::Operation<T> *> tmp_vars;

    /* copy the weights from each layer into _vars */
    for (vit = layers.begin(); vit != layers.end(); vit++) {
        tmp_vars = (*vit)->get_weights();
        this->_vars.insert(this->_vars.end(), tmp_vars.begin(), tmp_vars.end());
    }

    /* get pointers to network back and front operations */
    this->network_input_op_ptr = layers.front()->out();
    this->network_output_op_ptr = layers.back()->out();

    /* get pointers to network back and front tensors */
    this->network_input_tensor_ptr = this->network_input_op_ptr->get_output_tensor();
    this->network_output_tensor_ptr = this->network_output_op_ptr->get_output_tensor();

    /* init ground truth pointers */
    this->ground_truth_op_ptr =
        op::var<T>(this->_name + "::ground_truth", {params.batch_size, network_output_tensor_ptr->get_shape().back()},
                   {NONE, {}}, network_output_tensor_ptr->get_memory_type());
    this->ground_truth_tensor_ptr = this->ground_truth_op_ptr->get_output_tensor();

    /* init loss function -- _obj */
    switch (loss_func) {
        case optimizer::CROSS_ENTROPY:
            this->_obj = op::crossentropy(this->ground_truth_op_ptr, this->network_output_op_ptr);
            break;
        case optimizer::MSE:
            this->_obj = op::meansquarederror(this->ground_truth_op_ptr, this->network_output_op_ptr);
            break;
        default:
            std::fprintf(stderr, "Unknown loss function.\n");
            break;
    }
    this->_obj_tensor_ptr = this->_obj->get_output_tensor();
}

template <typename T>
NeuralNetwork<T>::~NeuralNetwork() {
    delete optim;
}

template <typename T>
magmadnn_error_t NeuralNetwork<T>::fit(Tensor<T> *x, Tensor<T> *y, metric_t &metric_out, bool verbose) {
    /* tensors to store on host -- we calculate accuracy and such on host */
    Tensor<T> *predicted, *actual, *host_network_output_tensor_ptr, *host_ground_truth_tensor_ptr;

    /* init the host tensors */
    predicted = new Tensor<T>({network_output_tensor_ptr->get_shape(0)}, {ZERO, {}},
                              HOST); /* this will store the result of the argmax on the output of the network */
    actual = new Tensor<T>({network_output_tensor_ptr->get_shape(0)}, {ZERO, {}},
                           HOST); /* this will store the result of the argmax on the ground_truth */
    host_network_output_tensor_ptr = new Tensor<T>(network_output_tensor_ptr->get_shape(), {NONE, {}},
                                                   HOST); /* used to move network output onto CPU */
    host_ground_truth_tensor_ptr =
        new Tensor<T>(ground_truth_tensor_ptr->get_shape(), {NONE, {}}, HOST); /* used to move ground_truth onto CPU */

    /* NULL check */
    if (this->_obj == NULL || this->optim == NULL) return (magmadnn_error_t) 1;

    /* Neural Network training Routine.
        1. Copy x tensor into input layer
        2. Forward propagate layer
        3. Call minimize on the optimizer
        4. Go back to 1
    */

    time_t start_time, end_time;
    magmadnn_error_t err = (magmadnn_error_t) 0;
    double cumulative_loss = 0.0;
    unsigned int n_samples = y->get_shape(0);
    unsigned int n_correct = 0;

    /* main training routine */
    time(&start_time);
    dataloader::LinearLoader<T> dataloader(x, y, this->model_params.batch_size);
    for (unsigned int i = 0; i < this->model_params.n_epochs; i++) {
        for (unsigned int j = 0; j < dataloader.get_num_batches(); j++) {
            /* load next batch into x and y */
            dataloader.next(this->network_input_tensor_ptr, this->ground_truth_tensor_ptr);

            /* forward pass */
            this->_obj->eval(true); /* forces evaluation */

            /* minimize using gradients */
            this->optim->minimize(this->_obj, this->_vars);

            /* get the argmax of the networks output (on CPU) */
            host_network_output_tensor_ptr->copy_from(*this->network_output_tensor_ptr);
            math::argmax(host_network_output_tensor_ptr, 0, predicted);

            /* get the argmax of the ground truth (on CPU) */
            host_ground_truth_tensor_ptr->copy_from(*this->ground_truth_tensor_ptr);
            math::argmax(host_ground_truth_tensor_ptr, 0, actual);

            /* update the accuracy and loss */
            for (unsigned int j = 0; j < this->model_params.batch_size; j++) {
                if (std::fabs(predicted->get(j) - actual->get(j)) <= 1E-8) {
                    n_correct++;
                }
            }
            this->_obj_tensor_ptr->get_memory_manager()->sync();
            cumulative_loss += this->_obj_tensor_ptr->get(0);
        }

        if (verbose) {
            printf("Epoch (%u/%u): accuracy=%.4g loss=%.4g time=%.4g\n", i, this->model_params.n_epochs,
                   n_correct / ((double) (i + 1) * n_samples),
                   cumulative_loss / ((double) (i + 1) * dataloader.get_num_batches()),
                   (double) time(NULL) - start_time);
        }

        /* resets dataloader for next epoch */
        dataloader.reset();
    }
    time(&end_time);

    /* update metrics */
    metric_out.accuracy = ((double) n_correct) / ((double) this->model_params.n_epochs * n_samples);
    metric_out.loss =
        ((double) cumulative_loss) / ((double) this->model_params.n_epochs * dataloader.get_num_batches());
    metric_out.training_time = (double) (end_time - start_time);

    if (verbose) {
        printf("Final Training Metrics: accuracy=%.4g loss=%.4g time=%.4g\n", metric_out.accuracy, metric_out.loss,
               metric_out.training_time);
    }

    /* free up any memory we used here */
    delete predicted;
    delete actual;
    delete host_network_output_tensor_ptr;
    delete host_ground_truth_tensor_ptr;

    return err;
}

template <typename T>
Tensor<T> *NeuralNetwork<T>::predict(Tensor<T> *sample) {
    this->network_input_tensor_ptr->copy_from(*sample, 0, sample->get_size());

    /* TODO only return the first row of output tensor */

    /* Forward propagate -- get the output tensor */
    return this->network_output_op_ptr->eval(true);
}

template <typename T>
unsigned int NeuralNetwork<T>::predict_class(Tensor<T> *sample) {
   // assert(T_IS_VECTOR(sample));

    /* copy sample into beginning of input tensor */
    this->network_input_tensor_ptr->copy_from(*sample, 0, sample->get_size());

    /* forward propagate network */
    this->network_output_op_ptr->eval(true);

    Tensor<T> output_tensor_host(this->network_output_tensor_ptr->get_shape(), {NONE, {}}, HOST);
    Tensor<T> argmax_tensor({output_tensor_host.get_shape(0)}, {NONE, {}}, HOST);

    output_tensor_host.copy_from(*this->network_output_tensor_ptr);

    math::argmax(&output_tensor_host, 0, &argmax_tensor);

    return argmax_tensor.get(0);
}

template <typename T>
void NeuralNetwork<T>::summary() {

    unsigned int name_w = 20, shape_w = 20, params_w = 16;
    
    std::cout << std::setw(name_w) << std::left << "Name";
    std::cout << std::setw(shape_w) << std::right << "Output Shape";
    std::cout << std::setw(params_w) << "# Params";
    std::cout << std::endl;
    std::cout << std::setfill('=') << std::setw(name_w + shape_w + params_w) << "";
    std::cout << std::endl << std::setfill(' ');


    for(int i = 0 ; i < this->layers.size(); i++){
        if(this->layers[i]->get_name() == "Activation")
            continue;

        std::cout << std::setw(name_w) << std::left << this->layers[i]->get_name();

        /*Make shape string*/
        std::vector<unsigned int> output_shape = this->layers[i]->get_output_shape();
        std::string shape = "(";
        for(int j = 0; j < output_shape.size() - 1; j++){
            shape += std::to_string(output_shape[j]) + ", "; 
        }
        shape += std::to_string(output_shape[output_shape.size() - 1]) + ")";

        std::cout << std::setw(shape_w) << std::right << shape;
        std::cout << std::setw(params_w) << this->layers[i]->get_num_params();
        std::cout << std::endl;
    }
}

template class NeuralNetwork<int>;
template class NeuralNetwork<float>;
template class NeuralNetwork<double>;

}  // namespace model
}  // namespace magmadnn
