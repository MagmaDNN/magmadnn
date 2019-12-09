/**
 * @file conv2dlayer.cpp
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-07-08
 *
 * @copyright Copyright (c) 2019
 */
#include <iostream>
#include "layer/conv2d/conv2dlayer.h"

namespace magmadnn {
namespace layer {

template <typename T>
Conv2dLayer<T>::Conv2dLayer(
      op::Operation<T>* input, const std::vector<unsigned int>& filter_shape, int out_channels,
      const std::vector<unsigned int>& padding, const std::vector<unsigned int>& strides,
      const std::vector<unsigned int>& dilation_rates, bool use_cross_correlation, bool use_bias,
      tensor_filler_t<T> filter_initializer, tensor_filler_t<T> bias_initializer)
    : Layer<T>::Layer(input->get_output_shape(), input),
      out_channels(out_channels),
      use_cross_correlation(use_cross_correlation),
      use_bias(use_bias),
      filter_initializer(filter_initializer),
      bias_initializer(bias_initializer) {
    if (padding.size() == 2) {
        pad_h = padding[0];
        pad_w = padding[1];
    } else {
        fprintf(stderr, "Error: Expected padding to have 2 elements\n");
    }

    if (strides.size() == 2) {
        stride_h = strides[0];
        stride_w = strides[1];
    } else {
        fprintf(stderr, "Error: Expected strides to have 2 elements\n");
    }

    if (dilation_rates.size() == 2) {
        dilation_h = dilation_rates[0];
        dilation_w = dilation_rates[1];
    } else {
        fprintf(stderr, "Error: Expected dilation_rates to have 2 elements\n");
    }

    init(filter_shape);
}

template <typename T>
Conv2dLayer<T>::~Conv2dLayer() {
    delete filter_tensor;
    if (use_bias) delete bias_tensor;
}

template <typename T>
std::vector<op::Operation<T>*> Conv2dLayer<T>::get_weights() {
    if (use_bias) {
        return {filter, bias};
    } else {
        return {filter};
    }
}

template <typename T>
void Conv2dLayer<T>::init(const std::vector<unsigned int>& filter_shape) {
    this->name = "Conv2d";

    assert(this->input_shape.size() == 4);
    assert(filter_shape.size() == 2);

    this->in_channels = this->input_shape[1];

    /* create filter tensor */
    this->filter_tensor =
        new Tensor<T>({static_cast<unsigned int>(this->out_channels), static_cast<unsigned int>(this->in_channels),
                       filter_shape[0], filter_shape[1]},
                      this->filter_initializer, this->input->get_memory_type());
    this->filter = op::var<T>("__" + this->name + "_layer_filter", this->filter_tensor);

    /* create the bias tensor if need be */
    if (use_bias) {
       /* TODO */
    }

    if (use_bias) {
       /* TODO */
    } else {
       this->output = op::conv2dforward(
             this->input, filter, pad_h, pad_w, stride_h, stride_w, dilation_h,
             dilation_w, use_cross_correlation);
    }
}

   // template <typename T>
   // std::size_t Conv2dLayer<T>::get_memory_size() const {
   //    return this->filter_tensor->get_memory_size();
   // }
   
template class Conv2dLayer<int>;
template class Conv2dLayer<float>;
template class Conv2dLayer<double>;

template <typename T>
Conv2dLayer<T>* conv2d(op::Operation<T>* input, const std::vector<unsigned int>& filter_shape, int out_channels,
                       const std::vector<unsigned int>& padding, const std::vector<unsigned int>& strides,
                       const std::vector<unsigned int>& dilation_rates, bool use_cross_correlation, bool use_bias,
                       tensor_filler_t<T> filter_initializer, tensor_filler_t<T> bias_initializer) {
    return new Conv2dLayer<T>(input, filter_shape, out_channels, padding, strides, dilation_rates,
                              use_cross_correlation, use_bias, filter_initializer, bias_initializer);
}
template Conv2dLayer<int>* conv2d(op::Operation<int>*, const std::vector<unsigned int>&, int,
                                  const std::vector<unsigned int>&, const std::vector<unsigned int>&,
                                  const std::vector<unsigned int>&, bool, bool, tensor_filler_t<int>,
                                  tensor_filler_t<int>);
template Conv2dLayer<float>* conv2d(op::Operation<float>*, const std::vector<unsigned int>&, int,
                                    const std::vector<unsigned int>&, const std::vector<unsigned int>&,
                                    const std::vector<unsigned int>&, bool, bool, tensor_filler_t<float>,
                                    tensor_filler_t<float>);
template Conv2dLayer<double>* conv2d(op::Operation<double>*, const std::vector<unsigned int>&, int,
                                     const std::vector<unsigned int>&, const std::vector<unsigned int>&,
                                     const std::vector<unsigned int>&, bool, bool, tensor_filler_t<double>,
                                     tensor_filler_t<double>);

template <typename T>
Conv2dLayer<T>* conv2d(op::Operation<T>* input, const std::vector<unsigned int>& filter_shape, int out_channels,
                       padding_t padding, const std::vector<unsigned int>& strides,
                       const std::vector<unsigned int>& dilation_rates, bool use_cross_correlation, bool use_bias,
                       tensor_filler_t<T> filter_initializer, tensor_filler_t<T> bias_initializer) {
    assert(strides.size() == 2 && filter_shape.size() == 2);

    /* Formula derived from
     * https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnGetConvolution2dForwardOutputDim
     */
    unsigned int padding_h, padding_w;

    if (padding == layer::SAME) {
        unsigned int tempval_h =
            (input->get_output_shape(2) - 1) * (strides[0] - 1) + (filter_shape[0] - 1) * dilation_rates[0] + 1;
        unsigned int tempval_w =
            (input->get_output_shape(3) - 1) * (strides[1] - 1) + (filter_shape[1] - 1) * dilation_rates[1] + 1;
        padding_h = tempval_h / 2;
        padding_w = tempval_w / 2;
    } else {
        padding_h = 0;
        padding_w = 0;
    }
    return new Conv2dLayer<T>(input, filter_shape, out_channels, {padding_h, padding_w}, strides, dilation_rates,
                              use_cross_correlation, use_bias, filter_initializer, bias_initializer);
}
template Conv2dLayer<int>* conv2d(op::Operation<int>*, const std::vector<unsigned int>&, int, padding_t,
                                  const std::vector<unsigned int>&, const std::vector<unsigned int>&, bool, bool,
                                  tensor_filler_t<int>, tensor_filler_t<int>);
template Conv2dLayer<float>* conv2d(op::Operation<float>*, const std::vector<unsigned int>&, int, padding_t,
                                    const std::vector<unsigned int>&, const std::vector<unsigned int>&, bool, bool,
                                    tensor_filler_t<float>, tensor_filler_t<float>);
template Conv2dLayer<double>* conv2d(op::Operation<double>*, const std::vector<unsigned int>&, int, padding_t,
                                     const std::vector<unsigned int>&, const std::vector<unsigned int>&, bool, bool,
                                     tensor_filler_t<double>, tensor_filler_t<double>);

}  // namespace layer
}  // namespace magmadnn
