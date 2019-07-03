/**
 * @file layer_utilities.h
 * @author Daniel Nichols
 * @version 0.1.0
 * @date 2019-07-03
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "tensor/tensor.h"
#include "compute/operation.h"
#include "layer/layer.h"

#include <cassert>


namespace magmadnn {
namespace layer {
namespace utilities {


template <typename T>
magmadnn_error_t copy_layer(layer::Layer<T> *dst, layer::Layer<T> *src);

template <typename T>
magmadnn_error_t copy_layers(std::vector<layer::Layer<T> *> dsts, std::vector<layer::Layer<T> *> srcs);

}
}
}