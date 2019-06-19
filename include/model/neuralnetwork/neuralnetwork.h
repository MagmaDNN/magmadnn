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
#include "model/model.h"
#include "layer/layers.h"
#include "optimizer/optimizers.h"
#include "math/argmax.h"

namespace magmadnn {
namespace model {

struct nn_params_t {
    unsigned int n_epochs;  /**<n_epochs number of epochs to train for */
    unsigned int batch_size;    /**<batch_size the size of the batch */
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
    NeuralNetwork(std::vector<layer::Layer<T> *> layers, optimizer::loss_t loss_func, optimizer::optimizer_t optimizer, nn_params_t params);

    /** Trains this neural network using x and y. The batch size and epochs are used from the model parameters given during
     * construction. If verbose, then training info is printed periodically.
     * @param x independent data
     * @param y ground truth <i>one-hot encoded</i> data
     * @param metric_out [out] metric to write training metrics in
     * @param verbose if true, run in verbose mode
     * @return magmadnn_error_t 0 on success
     */
    virtual magmadnn_error_t fit(Tensor<T> *x, Tensor<T> *y, metric_t& metric_out, bool verbose=false);
    virtual Tensor<T> *predict(Tensor<T> *sample);
    virtual unsigned int predict_class(Tensor<T> *sample);

protected:
    typename std::vector<layer::Layer<T> *> layers;
    optimizer::loss_t loss_func;
    optimizer::optimizer_t optimizer;
    nn_params_t model_params;

    T default_learning_rate = (T) 0.05;
    std::vector<op::Operation<T> *> _vars;
    op::Operation<T> *_obj;
};

}   // namespace model
}   // namespace magmadnn