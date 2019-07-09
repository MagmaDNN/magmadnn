/**
 * @file neuralnetwork_utilities.cpp
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-07-03
 * 
 * @copyright Copyright (c) 2019
 */
#include "model/neuralnetwork/neuralnetwork_utilities.h"

namespace magmadnn {
namespace model {
namespace utilities {


template <typename T>
magmadnn_error_t copy_network(model::NeuralNetwork<T>& dst, model::NeuralNetwork<T>& src) {

    assert( dst.get_layers().size() == src.get_layers().size() );

    return ::magmadnn::layer::utilities::copy_layers(dst.get_layers(), src.get_layers());
}
template magmadnn_error_t copy_network(model::NeuralNetwork<int>& dst, model::NeuralNetwork<int>& src);
template magmadnn_error_t copy_network(model::NeuralNetwork<float>& dst, model::NeuralNetwork<float>& src);
template magmadnn_error_t copy_network(model::NeuralNetwork<double>& dst, model::NeuralNetwork<double>& src);


}
}
}