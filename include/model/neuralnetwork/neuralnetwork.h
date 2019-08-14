/**
 * @file neuralnetwork.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-30
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include <cmath>
#include <ctime>
#include "compute/op_utilities.h"
#include "dataloader/dataloaders.h"
#include "layer/layers.h"
#include "math/argmax.h"
#include "model/model.h"
#include "optimizer/optimizers.h"

namespace magmadnn {
namespace model {

struct nn_params_t {
    unsigned int n_epochs;        /**<n_epochs number of epochs to train for */
    unsigned int batch_size;      /**<batch_size the size of the batch */
    double learning_rate;         /**<initial learning rate */
    double momentum = 0.9;        /**<momentum rate */
    double decaying_factor = 0.9; /**<decaying factor for RMSProp */
    double beta1 = 0.9;           /**<beta1 for Adam */
    double beta2 = 0.999;         /**<beta2 for Adam */
};

template <typename T>
class NeuralNetwork : public Model<T> {
   public:
    /** Constructs a neural network with the given parameters.
     * @param layers a vector of the layers comprising the network.
     * @param loss_func a loss_t enumerant describing the loss function to use at the end of the network.
     * @param optimizer a optimizer_t enumerant describing the optimizer to use in training.
     * @param params a set of neural network parameters.
     */
    NeuralNetwork(std::vector<layer::Layer<T> *> layers, optimizer::loss_t loss_func, optimizer::optimizer_t optimizer,
                  nn_params_t params);

    /** Constructs a neural network. Allows you to supply custom optimizer.
     * @param layers a vector of the layers comprising the network.
     * @param loss_func a loss_t enumerant describing the loss function to use at the end of the network.
     * @param optim Optimizer<T> that will be used to optimize the network
     * @param params a set of neural network parameters.
     */
    NeuralNetwork(std::vector<layer::Layer<T> *> layers, optimizer::loss_t loss_func, optimizer::Optimizer<T> *optim,
                  nn_params_t params);

    virtual ~NeuralNetwork();

    /** Trains this neural network using x and y. The batch size and epochs are used from the model parameters given
     * during construction. If verbose, then training info is printed periodically.
     * @param x independent data
     * @param y ground truth <i>one-hot encoded</i> data
     * @param metric_out [out] metric to write training metrics in
     * @param verbose if true, run in verbose mode
     * @return magmadnn_error_t 0 on success
     */
    virtual magmadnn_error_t fit(Tensor<T> *x, Tensor<T> *y, metric_t &metric_out, bool verbose = false);
    virtual Tensor<T> *predict(Tensor<T> *sample);
    virtual unsigned int predict_class(Tensor<T> *sample);

    virtual std::vector<layer::Layer<T> *> get_layers() { return this->layers; }

   protected:
    typename std::vector<layer::Layer<T> *> layers;
    optimizer::loss_t loss_func;
    optimizer::optimizer_t optimizer;
    nn_params_t model_params;

    /* network inputs/output */
    op::Operation<T> *network_input_op_ptr, *network_output_op_ptr;
    Tensor<T> *network_input_tensor_ptr, *network_output_tensor_ptr;

    /* ground truth */
    op::Operation<T> *ground_truth_op_ptr;
    Tensor<T> *ground_truth_tensor_ptr;

    T default_learning_rate = (T) 0.05;    /* assumed learning rate if one is not given */
    std::vector<op::Operation<T> *> _vars; /* network weights */
    op::Operation<T> *_obj;                /* objective function to optimize -- i.e. the loss function */
    Tensor<T> *_obj_tensor_ptr;            /* pointer to objective function's tensor */
    optimizer::Optimizer<T> *optim;        /* network optimizer */
};

}  // namespace model
}  // namespace magmadnn