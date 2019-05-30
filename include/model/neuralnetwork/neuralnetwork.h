/**
 * @file neuralnetwork.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-30
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "model/model.h"
#include "layer/layers.h"
#include "optimizer/optimizers.h"

namespace magmadnn {
namespace model {

struct nn_params_t {
    unsigned int n_epochs;
    unsigned int batch_size;
};

template <typename T>
class NeuralNetwork : public Model<T> {
public:
    NeuralNetwork(std::vector<layer::Layer<T> *> layers, optimizer::loss_t loss_func, optimizer::Optimizer<T> *optim, nn_params_t params);
    NeuralNetwork(std::vector<layer::Layer<T> *> layers, optimizer::loss_t loss_func, optimizer::optimizer_t optimizer, nn_params_t params);

    virtual magmadnn_error_t fit(Tensor<T> *x, Tensor<T> *y, metric_t& metric_out, bool verbose=false);
    virtual Tensor<T> *predict(Tensor<T> *sample);
    virtual unsigned int predict_class(Tensor<T> *sample);

protected:
    typename std::vector<layer::Layer<T> *> layers;
    optimizer::loss_t loss_func;
    optimizer::Optimizer<T> *optimizer;
    nn_params_t model_params;

    T default_learning_rate = (T) 0.05;

    op::Operation<T> *_obj;
};

}   // namespace model
}   // namespace magmadnn