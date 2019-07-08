/**
 * @file conv2dlayer.h
 * @author Daniel Nichols
 * @version 0.1.0
 * @date 2019-07-08
 * 
 * @copyright Copyright (c) 2019
 */
#include <vector>
#include "layer/layer.h"
#include "tensor/tensor.h"
#include "compute/operation.h"
#include "compute/variable.h"
#include "compute/conv2dforward/conv2dforwardop.h"

namespace magmadnn {
namespace layer {

template <typename T>
class Conv2dLayer : public Layer<T> {
public:
    Conv2dLayer(op::Operation<T> *input, const std::vector<unsigned int>& filter_shape={3, 3}, int out_channels=1, const std::vector<unsigned int>& padding={0,0},
        const std::vector<unsigned int>& strides={1,1}, const std::vector<unsigned int>& dilation_rates={1,1}, bool use_cross_correlation=true, bool use_bias=false,
        tensor_filler_t<T> filter_initializer={GLOROT,{0.0,0.2f}}, tensor_filler_t<T> bias_initializer={GLOROT,{0.0,0.2f}});

    virtual ~Conv2dLayer();


    virtual std::vector<op::Operation<T> *> get_weights();

    op::Operation<T> *get_filter() { return filter; }
    op::Operation<T> *get_bias() { return bias; }

protected:
    void init(const std::vector<unsigned int>& filter_shape);

    Tensor<T> *filter_tensor;
    Tensor<T> *bias_tensor;

    op::Operation<T> *filter;
    op::Operation<T> *bias;

    int in_channels, out_channels;
    bool use_cross_correlation, use_bias;
    int pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w;

    tensor_filler_t<T> filter_initializer, bias_initializer;

};


/** Create a convolutional 2d Operation.
 * @tparam T numeric
 * @param input the input data; must be a 4D tensor in format NCHW (N-batch, C-Channel, H-height, W-Width)
 * @param filter_shape 
 * @param out_channels 
 * @param padding 
 * @param strides 
 * @param dilation_rates 
 * @param use_cross_correlation 
 * @param use_bias 
 * @param filter_initializer 
 * @param bias_initializer 
 * @return Conv2dLayer<T>* 
 */
template <typename T>
Conv2dLayer<T>* conv2d(op::Operation<T> *input, const std::vector<unsigned int>& filter_shape={3, 3}, int out_channels=1, const std::vector<unsigned int>& padding={0,0},
        const std::vector<unsigned int>& strides={1,1}, const std::vector<unsigned int>& dilation_rates={1,1}, bool use_cross_correlation=true, bool use_bias=false,
        tensor_filler_t<T> filter_initializer={GLOROT,{0.0,0.2f}}, tensor_filler_t<T> bias_initializer={GLOROT,{0.0,0.2f}});

}   // layer
}   // magmadnn