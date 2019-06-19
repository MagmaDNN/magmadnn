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
    double accuracy;    /**<accuracy the training accuracy of the model */
    double loss;        /**<loss the final loss from the models loss function */
    double training_time;   /**<training_time the training duration in seconds */
};

template <typename T>
class Model {
public:
    /** Creates a new model.
     */
    Model() {}

    /** train this model on x and y. 
     * @param x model input data
     * @param y ground truth data
     * @param metric_out [out] a metric_t struct in which the model run metrics are stored.
     * @param verbose if true then training info is printed every couple iterations
     * @return magmadnn_error_t 0 on success; non-zero otherwise
     */
    virtual magmadnn_error_t fit(Tensor<T> *x, Tensor<T> *y, metric_t& metric_out, bool verbose=false) = 0;

    /** return a one-hot encoded output of the network on this sample
     * @param sample a single sample 
     * @return Tensor<T>* one-hot encoded prediction
     */
    virtual Tensor<T> *predict(Tensor<T> *sample) = 0;

    /** This is equivalent to math::argmax(predict(sample)), where the one-hot encoded predict value is used.
     * @param sample 
     * @return unsigned int the index of the predicted class.
     */
    virtual unsigned int predict_class(Tensor<T> *sample) = 0;

    /** Gets the training accuracy of the last run.
     * @return double training accuracy of last run.
     */
    virtual double get_accuracy() { return _last_training_metric.accuracy; }

    /** Gets the loss of the last run.
     * @return double loss of last run.
     */
    virtual double get_loss() { return _last_training_metric.loss; }

    /** Gets training time of last run.
     * @return double training time of last run.
     */
    virtual double get_training_time() { return _last_training_metric.training_time; }

    /** name of this model
     * @return std::string 
     */
    virtual std::string get_name() { return this->_name; }

protected:
    std::string _name;
    metric_t _last_training_metric;

};

}   // namespace model
}   // namespace magmadnn

