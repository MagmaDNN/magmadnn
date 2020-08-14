/**
 * @file alexnet.h
 * @author Florent Lopez
 * @version 1.0
 * @date 2020-07-30
 *
 * @copyright Copyright (c) 2020
 */
#pragma once

#include "magmadnn.h"

namespace magmadnn {

template <typename T>
class Alexnet {

public:
   static std::vector<layer::Layer<T> *> alexnet(
         op::Variable<T>* x_batch, int nclasses) {

      /* std::vector<layer::Layer<T> *> layers; */
      auto input = layer::input<T>(x_batch);
      
      auto conv2d1 = layer::conv2d<T>(input->out(), {11, 11}, 64, {2, 2}, {4, 4}, {1, 1});
      auto act1 = layer::activation<T>(conv2d1->out(), layer::RELU);
      auto pool1 = layer::pooling<T>(act1->out(), {3, 3}, {0, 0}, {2, 2}, AVERAGE_POOL);

      auto conv2d2 = layer::conv2d<T>(pool1->out(), {5, 5}, 192, layer::SAME, {1, 1}, {1, 1});
      auto act2 = layer::activation<T>(conv2d2->out(), layer::RELU);
      auto pool2 = layer::pooling<T>(act2->out(), {3, 3}, {0, 0}, {2, 2}, AVERAGE_POOL);

      auto conv2d3 = layer::conv2d<T>(pool2->out(), {3, 3}, 384, layer::SAME, {1, 1}, {1, 1});
      auto act3 = layer::activation<T>(conv2d3->out(), layer::RELU);

      auto conv2d4 = layer::conv2d<T>(act3->out(), {3, 3}, 384, layer::SAME, {1, 1}, {1, 1});
      auto act4 = layer::activation<T>(conv2d4->out(), layer::RELU);

      auto conv2d5 = layer::conv2d<T>(act4->out(), {3, 3}, 256, layer::SAME, {1, 1}, {1, 1});
      auto act5 = layer::activation<T>(conv2d5->out(), layer::RELU);

      auto pool3 = layer::pooling<T>(act5->out(), {3, 3}, layer::SAME, {2, 2}, AVERAGE_POOL);

      auto dropout1 = layer::dropout<float>(pool3->out(), 0.5);

      auto flatten = layer::flatten<T>(dropout1->out());

      auto fc1 = layer::fullyconnected<T>(flatten->out(), 4096, true);
      auto act6 = layer::activation<T>(fc1->out(), layer::RELU);

      auto fc2 = layer::fullyconnected<T>(act6->out(), 4096, true);
      auto act7 = layer::activation<T>(fc2->out(), layer::RELU);

      auto fc3 = layer::fullyconnected<T>(act7->out(), nclasses, false);
      auto act8 = layer::activation<T>(fc3->out(), layer::SOFTMAX);

      auto output = layer::output<T>(act8->out());

      std::vector<layer::Layer<T> *> layers =
         {input,
          conv2d1, act1, pool1,
          conv2d2, act2, pool2,
          conv2d3, act3, 
          conv2d4, act4, 
          conv2d5, act5,
          pool3,
          dropout1,
          flatten,
          fc1, act6,
          fc2, act7,
          fc3, act8,
          output};
      
      return layers;
   }

}; // Class Alexnet

} // End of magmadnn namespace
