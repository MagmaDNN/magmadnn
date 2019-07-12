/**
 * @file inputlayer.h
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-02-26
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include <vector>
#include "compute/operation.h"
#include "layer/layer.h"
#include "tensor/tensor.h"

namespace magmadnn {
namespace layer {

template <typename T>
class InputLayer : public Layer<T> {
   public:
    InputLayer(op::Operation<T> *input);

    virtual std::vector<op::Operation<T> *> get_weights();

   protected:
    void init();
};

template <typename T>
InputLayer<T> *input(op::Operation<T> *input);

}  // namespace layer
}  // namespace magmadnn