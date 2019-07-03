/**
 * @file neuralnetwork_utilities.cpp
 * @author Daniel Nichols
 * @version 0.1.0
 * @date 2019-07-03
 * 
 * @copyright Copyright (c) 2019
 */
#include "model/neuralnetwork/neuralnetwork_utilities.h"

namespace magmadnn {
namespace model {
namespace utilities {


template <typename T>
magmadnn_error_t copy_network(model::NeuralNetwork<T>& dst, const model::NeuralNetwork<T>& src) {
    const std::vector<::magmadnn::layer::Layer<T> *>& dst_layers = dst.get_layers();
    const std::vector<::magmadnn::layer::Layer<T> *>& src_layers = src.get_layers();

    assert( dst_layers.size() == dst_layers.size() );

    ::magmadnn::layer::utilities::copy_layers(dst_layers, src_layers);
}


}
}
}