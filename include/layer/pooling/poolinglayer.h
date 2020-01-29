/**
 * @file poolinglayer.h
 * @author Sedrick Keh
 * @version 1.0
 * @date 2019-07-08
 *
 * @copyright Copyright (c) 2019
 */
#include <vector>
#include "compute/operation.h"
#include "compute/tensor_operations.h"
#include "layer/conv2d/conv2dlayer.h"
#include "layer/layer.h"
#include "tensor/tensor.h"

namespace magmadnn {
namespace layer {

template <typename T>
class PoolingLayer : public Layer<T> {
   public:
    PoolingLayer(op::Operation<T>* input, const std::vector<unsigned int>& filter_shape = {2, 2},
                 const std::vector<unsigned int>& padding = {0, 0}, const std::vector<unsigned int>& strides = {1, 1},
                 const std::vector<unsigned int>& dilation_rates = {1, 1}, pooling_mode mode = MAX_POOL,
                 bool propagate_nan = false);

    virtual ~PoolingLayer();

    virtual std::vector<op::Operation<T>*> get_weights();

   protected:
    void init();

    pooling_mode mode;
    bool propagate_nan;
    int filter_h, filter_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w;
};

/** A new Pooling2d layer.
 * @tparam T numeric
 * @param input input, usually output of an activated convolutional layer
 * @param filter_shape shape to filter image
 * @param padding how to pad pooling
 * @param strides striding
 * @param dilation_rates rate of dilation. must be a two dimensional vector; defaults to {1,1}
 * @param mode MAX_POOL or AVERAGE_POOL
 * @param propagate_nan propagate nan values
 * @return PoolingLayer<T>* a pooling layer
 */
template <typename T>
PoolingLayer<T>* pooling(op::Operation<T>* input, const std::vector<unsigned int>& filter_shape = {2, 2},
                         const std::vector<unsigned int>& padding = {0, 0},
                         const std::vector<unsigned int>& strides = {1, 1},
                         const std::vector<unsigned int>& dilation_rates = {1, 1}, pooling_mode mode = MAX_POOL,
                         bool propagate_nan = false);

/* Indicate pooling type when creating */
template <typename T>
PoolingLayer<T>* pooling(op::Operation<T>* input, const std::vector<unsigned int>& filter_shape = {2, 2},
                         layer::padding_t padding = layer::SAME, const std::vector<unsigned int>& strides = {1, 1},
                         const std::vector<unsigned int>& dilation_rates = {1, 1}, pooling_mode mode = MAX_POOL,
                         bool propagate_nan = false);

}  // namespace layer
}  // namespace magmadnn