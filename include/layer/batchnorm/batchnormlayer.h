/**
 * @file batchnormlayer.h
 * @author Sedrick Keh
 * @version 1.0
 * @date 2019-07-24
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include <vector>
#include "compute/batchnorm/batchnormop.h"
#include "compute/operation.h"
#include "compute/tensor_operations.h"
#include "layer/layer.h"
#include "tensor/tensor.h"

namespace magmadnn {
namespace layer {

template <typename T>
class BatchNormLayer : public Layer<T> {
   public:
    BatchNormLayer(op::Operation<T> *input);
    virtual ~BatchNormLayer();

    virtual std::vector<op::Operation<T> *> get_weights();

   protected:
    void init();
};

/** A new Batch Normalization layer.
 * @tparam T numeric
 * @param input input tensor to normalize.
 * @return BatchNormLayer<T>* a batchnorm layer.
 */
template <typename T>
BatchNormLayer<T> *batchnorm(op::Operation<T> *input);

}  // namespace layer
}  // namespace magmadnn