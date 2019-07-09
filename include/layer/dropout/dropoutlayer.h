/**
 * @file dropoutlayer.h
 * @author Sedrick Keh
 * @version 1.0
 * @date 2019-06-28
 * 
 * @copyright Copyright (c) 2019
 */
#include <vector>
#include "layer/layer.h"
#include "tensor/tensor.h"
#include "compute/operation.h"
#include "compute/tensor_operations.h"

const float DEFAULT_DROPOUT_RATE = 0.2;
const unsigned long long DEFAULT_SEED = time(NULL);

namespace magmadnn {
namespace layer {

template <typename T>
class DropoutLayer : public Layer<T> {
public:
    DropoutLayer(op::Operation<T> *input, float dropout_rate, unsigned long long seed);
    virtual ~DropoutLayer();

    virtual std::vector<op::Operation<T> *> get_weights();

protected:
    void init();

    float dropout_rate;
    unsigned long long seed;
};

/** A new Dropout layer.
 * @tparam T numeric
 * @param input input tensor to randomly dropout.
 * @param dropout_rate percentage of values to dropout.
 * @param seed seed for random sampling.
 * @return DropoutLayer<T>* a dropout layer.
 */
template <typename T>
DropoutLayer<T>* dropout(op::Operation<T> *input, float dropout_rate = DEFAULT_DROPOUT_RATE, unsigned long long seed = DEFAULT_SEED);

}   // layer
}   // magmadnn