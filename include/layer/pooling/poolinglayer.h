/**
 * @file poolinglayer.h
 * @author Sedrick Keh
 * @version 0.0.1
 * @date 2019-07-08
 * 
 * @copyright Copyright (c) 2019
 */
#include <vector>
#include "layer/layer.h"
#include "tensor/tensor.h"
#include "compute/operation.h"
#include "compute/tensor_operations.h"

namespace magmadnn {
namespace layer {

template <typename T>
class PoolingLayer : public Layer<T> {
public:
    PoolingLayer(op::Operation<T> *input, const std::vector<unsigned int>& filter_shape={2, 2}, const std::vector<unsigned int>& padding={0,0},
        const std::vector<unsigned int>& strides={1,1}, pooling_mode mode = MAX_POOL, bool propagate_nan=false);

    virtual ~PoolingLayer();

    virtual std::vector<op::Operation<T> *> get_weights();

protected:
    void init();

    pooling_mode mode;
    bool propagate_nan;
    int filter_h, filter_w, pad_h, pad_w, stride_h, stride_w;

};

template <typename T>
PoolingLayer<T>* pooling(op::Operation<T> *input, const std::vector<unsigned int>& filter_shape={2, 2}, const std::vector<unsigned int>& padding={0,0},
        const std::vector<unsigned int>& strides={1,1}, pooling_mode mode = MAX_POOL, bool propagate_nan=false);

}   // layer
}   // magmadnn