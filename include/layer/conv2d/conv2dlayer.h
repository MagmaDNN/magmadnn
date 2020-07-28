/**
 * @file conv2dlayer.h
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-07-08
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include <vector>
#include "compute/conv2dforward/conv2dforwardop.h"
#include "compute/operation.h"
#include "compute/variable.h"
#include "layer/layer.h"
#include "tensor/tensor.h"

namespace magmadnn {
namespace layer {

   enum padding_t { SAME, VALID };

   template <typename T>
   class Conv2dLayer : public Layer<T> {
   public:
      Conv2dLayer(
            op::Operation<T>* input,
            const std::vector<unsigned int>& filter_shape = {3, 3},
            int out_channels = 1,
            const std::vector<unsigned int>& padding = {0, 0},
            const std::vector<unsigned int>& strides = {1, 1},
            const std::vector<unsigned int>& dilation_rates = {1, 1},
            bool use_cross_correlation = true,
            bool use_bias = false, tensor_filler_t<T> filter_initializer = {GLOROT, {0.0, 0.2f}},
            tensor_filler_t<T> bias_initializer = {GLOROT, {0.0, 0.2f}});

      virtual ~Conv2dLayer();

      virtual std::vector<op::Operation<T>*> get_weights();

      virtual unsigned int get_num_params();

      op::Operation<T>* get_filter() { return filter; }
      op::Operation<T>* get_bias() { return bias; }

      std::size_t get_memory_size() const {
         return this->filter_tensor->get_memory_size();
      }
      
   protected:
      void init(const std::vector<unsigned int>& filter_shape);

      Tensor<T>* filter_tensor;
      Tensor<T>* bias_tensor;

      op::Operation<T>* filter;
      op::Operation<T>* bias;

      int in_channels, out_channels;
      bool use_cross_correlation, use_bias;
      int pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w;

      tensor_filler_t<T> filter_initializer, bias_initializer;
   };

/** Create a Conv2d Layer. Computes the output on the GPU. Uses CuDNN. No host functionality.
 * @tparam T numeric
 * @param input the input data; must be a 4D tensor in format NCHW (N-batch, C-Channel, H-height, W-Width)
 * @param filter_shape shape of the convolution kernel. must be a 2 dimensional vector; defaults to {3,3}
 * @param out_channels number of output filters; defaults to 1
 * @param padding the padding size to use. must be a two dimensional vector; defaults to {0,0}
 * @param strides striding of convolution. must be a two dimensional vector; defaults to {1,1}
 * @param dilation_rates rate of dilation. must be a two dimensional vector; defaults to {1,1}
 * @param use_cross_correlation whether to do a cross correlation convolution or standard convolution; defaults to cross
 * correlation
 * @param use_bias use convolutional bias or not; defaults to false
 * @param filter_initializer how to initialize the filter tensor; defaults to {GLOROT,{0.0f,0.2f}}
 * @param bias_initializer how to initialize the bias tensor; defaults to {GLOROT, {0.0f,0.2f}}
 * @return Conv2dLayer<T>* a new layer
 */
template <typename T>
Conv2dLayer<T>* conv2d(op::Operation<T>* input, const std::vector<unsigned int>& filter_shape = {3, 3},
                       int out_channels = 1, const std::vector<unsigned int>& padding = {0, 0},
                       const std::vector<unsigned int>& strides = {1, 1},
                       const std::vector<unsigned int>& dilation_rates = {1, 1}, bool use_cross_correlation = true,
                       bool use_bias = false, tensor_filler_t<T> filter_initializer = {GLOROT, {0.0, 0.2f}},
                       tensor_filler_t<T> bias_initializer = {GLOROT, {0.0, 0.2f}});

/* Indicate pooling type when creating */
template <typename T>
Conv2dLayer<T>* conv2d(op::Operation<T>* input, const std::vector<unsigned int>& filter_shape = {3, 3},
                       int out_channels = 1, layer::padding_t padding = layer::SAME,
                       const std::vector<unsigned int>& strides = {1, 1},
                       const std::vector<unsigned int>& dilation_rates = {1, 1}, bool use_cross_correlation = true,
                       bool use_bias = false, tensor_filler_t<T> filter_initializer = {GLOROT, {0.0, 0.2f}},
                       tensor_filler_t<T> bias_initializer = {GLOROT, {0.0, 0.2f}});

}  // namespace layer
}  // namespace magmadnn
