/**
 * @file resnet_cifar10.cpp
 * @author Florent Lopez
 * @version 1.0
 * @date 2020-07-30
 *
 * @copyright Copyright (c) 2020
 */
#include "magmadnn.h"

namespace magmadnn {

template <typename T>
std::vector<layer::Layer<T> *> basic_block(
      op::Operation<T>* input, int channels,
      const std::vector<unsigned int> strides,
      bool enable_shorcut = true) {

   std::vector<layer::Layer<T> *> layers;

   // In the resnet model, downsampling is achieved in the
   // convolutions by using a stride > 1.
   bool downsample = (strides[0] > 1);
   
   // std::cout << "[basic_block]"
   //           << " input size = " << input->get_output_shape(0)
   //           << ", " << input->get_output_shape(1)
   //           << ", " << input->get_output_shape(2)
   //           << ", " << input->get_output_shape(3)
   //           << std::endl;

   auto conv2d1 = layer::conv2d<T>(input, {3, 3}, channels, {1, 1}, strides, {1, 1});
   auto bn1 = layer::batchnorm(conv2d1->out());
   auto act1 = layer::activation<T>(bn1->out(), layer::RELU);

   auto conv2d2 = layer::conv2d<T>(act1->out(), {3, 3}, channels, {1, 1}, {1, 1}, {1, 1});
   auto bn2 = layer::batchnorm(conv2d2->out());

   // auto shortcut = op::pow(input, 2, true, false);
   // auto shortcut = op::pow(conv2d2->out(), 1, true, false);
   // auto shortcut = op::add(input, conv2d2->out(), true, false);
   // auto shortcut = layer::shortcut(conv2d2->out(), input);

   // std::cout << "[basic_block]"
   //           << " conv2 output size = " << bn2->out()->get_output_shape(0)
   //           << ", " << bn2->out()->get_output_shape(1)
   //           << ", " << bn2->out()->get_output_shape(2)
   //           << ", " << bn2->out()->get_output_shape(3)
   //           << std::endl;

   layers.insert(std::end(layers), conv2d1);
   layers.insert(std::end(layers), bn1);
   layers.insert(std::end(layers), act1);

   layers.insert(std::end(layers), conv2d2);
   layers.insert(std::end(layers), bn2);
   
   layer::Layer<T> *act2 = nullptr;
   
   if (enable_shorcut) {
      // Residual layer
      if (downsample) {
         // Downsampling

         auto downsample_conv2d = layer::conv2d<T>(input, {1, 1}, channels, {0, 0}, strides, {1, 1});
         // auto shortcut = op::add(bn2->out(), downsample_conv2d->out());
         auto downsample_bn = layer::batchnorm(downsample_conv2d->out());
         auto shortcut = op::add(bn2->out(), downsample_bn->out());

         // auto downsample_pool = layer::pooling<T>(input, {1, 1}, {0, 0}, strides, MAX_POOL);

         // std::cout << "[basic_block]"
         //           << " downsample_pool output size = " << downsample_pool->out()->get_output_shape(0)
         //           << ", " << downsample_pool->out()->get_output_shape(1)
         //           << ", " << downsample_pool->out()->get_output_shape(2)
         //           << ", " << downsample_pool->out()->get_output_shape(3)
         //           << std::endl;

         // auto shortcut = op::add(bn2->out(), downsample_pool->out());
         
         // std::cout << "[basic_block]"
         //           << " downsample_conv2d output size = " << downsample_conv2d->out()->get_output_shape(0)
         //           << ", " << downsample_conv2d->out()->get_output_shape(1)
         //           << ", " << downsample_conv2d->out()->get_output_shape(2)
         //           << ", " << downsample_conv2d->out()->get_output_shape(3)
         //           << std::endl;

         // layers.insert(std::end(layers), downsample_conv2d);
         // layers.insert(std::end(layers), downsample_bn);

         act2 = layer::activation<T>(shortcut, layer::RELU);

         // act2 = layer::activation<T>(bn2->out(), layer::RELU);

      }
      else {
         auto shortcut = op::add(bn2->out(), input);

         // auto act2 = layer::activation<T>(conv2d2->out(), layer::RELU);
         // auto act2 = layer::activation<T>(shortcut->out(), layer::RELU);
         act2 = layer::activation<T>(shortcut, layer::RELU);
      }
   }
   else {
      act2 = layer::activation<T>(bn2->out(), layer::RELU);
   }

   layers.insert(std::end(layers), act2);

   // std::vector<layer::Layer<T> *> layers =
   //    {conv2d1, bn1, act1,
   //     conv2d2, bn2,
   //     // shortcut,
   //     act2};

   return layers;
}

   
} // End of magmadnn namespace
