/**
 * @file model.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-30
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once


#include <string>
#include "types.h"
#include "tensor/tensor.h"
#include "compute/operation.h"
#include "optimizer/optimizer.h"


namespace magmadnn {
namespace model {

struct metric_t {
    double accuracy;
    double loss;
    double training_time;
};

template <typename T>
class Model {
public:
    Model() {}

    virtual magmadnn_error_t fit(Tensor<T> *x, Tensor<T> *y, metric_t& metric_out, bool verbose=false) = 0;
    virtual Tensor<T> *predict(Tensor<T> *sample) = 0;
    virtual unsigned int predict_class(Tensor<T> *sample) = 0;

    virtual double get_accuracy() { return _last_training_metric.accuracy; }
    virtual double get_loss() { return _last_training_metric.loss; }
    virtual double get_training_time() { return _last_training_metric.training_time; }

    virtual std::string get_name() { return this->_name; }

protected:
    std::string _name;
    metric_t _last_training_metric;

};

}   // namespace model
}   // namespace magmadnn

