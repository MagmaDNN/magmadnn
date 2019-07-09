/**
 * @file neuralnetwork_utilities.h
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-07-03
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "tensor/tensor.h"
#include "layer/layer.h"
#include "model/neuralnetwork/neuralnetwork.h"
#include "layer/layer_utilities.h"

namespace magmadnn {
namespace model {
namespace utilities {


template <typename T>
magmadnn_error_t copy_network(model::NeuralNetwork<T>& dst, model::NeuralNetwork<T>& src);


}
}
}