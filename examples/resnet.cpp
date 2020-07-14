/**
 * @file resnet.cpp
 * @author Florent Lopez
 * @version 1.0
 * @date 2020-07-14
 *
 * @copyright Copyright (c) 2020
 */

#include "magmadnn.h"

#include <iostream>

using namespace magmadnn;


template <typename T>
std::vector<layer::Layer<T> *> basic_block(
      op::Operation<T>* input, int channels,
      const std::vector<unsigned int>& strides) {

   auto conv2d1 = layer::conv2d<T>(input, {3, 3}, channels, {1, 1}, strides, {1, 1});
   // TODO batchnorm
   auto act1 = layer::activation<T>(conv2d1->out(), layer::RELU);

   auto conv2d2 = layer::conv2d<T>(act1->out(), {3, 3}, channels, {1, 1}, {1, 1}, {1, 1});
   // TODO batchnorm

   auto act2 = layer::activation<T>(op::add(input, conv2d1->out()), layer::RELU);

   std::vector<layer::Layer<T> *> layers =
      {conv2d1, act1,
       conv2d2, act2};

   return layers;
}

int main(int argc, char** argv) {

   std::string context = "resnet";
   
   // Data type
   using T = float;

   magmadnn_init();

   // Location of the CIFAR-10 dataset
   std::string const cifar10_dir = ".";
   // Location of the CIFAR-100 dataset
   std::string const cifar100_dir = ".";
   // Load CIFAR-10 trainnig dataset
   magmadnn::data::CIFAR10<T> train_set(cifar10_dir, magmadnn::data::Train);
   magmadnn::data::CIFAR10<T> test_set(cifar10_dir, magmadnn::data::Test);
   // Load CIFAR-100 trainnig dataset   
   // magmadnn::data::CIFAR100<T> train_set(cifar100_dir, magmadnn::data::Train);
   // magmadnn::data::CIFAR100<T> test_set(cifar100_dir, magmadnn::data::Test);

   // Training parameters
   magmadnn::model::nn_params_t params;
   params.batch_size = 128;
   params.n_epochs = 500;
   // params.learning_rate = 0.05;
   // params.learning_rate = 0.01;
   params.learning_rate = 0.001;
   // params.learning_rate = 0.002;
   // params.learning_rate = 0.0001;
   // params.learning_rate = 1.0;
   // params.decaying_factor = 0.99;
   
   // Memory
   magmadnn::memory_t training_memory_type;
#if defined(MAGMADNN_HAVE_CUDA)
   int devid = 0;
   cudaSetDevice(1);
   cudaGetDevice(&devid);
   std::cout << "[" << context << "] GPU training (" << devid << ")" << std::endl;
   training_memory_type = DEVICE;
#else
   training_memory_type = HOST;
#endif

   std::cout << "[" << context << "] Image dimensions: " << train_set.nrows() << " x " << train_set.ncols() << std::endl;
   std::cout << "[" << context << "] Number of chanels: " << train_set.nchanels() << std::endl;
   std::cout << "[" << context << "] Number of classes: " << train_set.nclasses() << std::endl;
   std::cout << "[" << context << "] Training set size: " << train_set.nimages() << std::endl;
   
   auto x_batch = op::var<T>(
         "x_batch",
         {params.batch_size, train_set.nchanels(),  train_set.nrows(), train_set.ncols()},
         {NONE, {}},
         training_memory_type);

   auto input = layer::input<T>(x_batch);

   auto conv2d1 = layer::conv2d<T>(input->out(), {7, 7}, 64, {3, 3}, {2, 2}, {1, 1});

   // auto conv2d1 = layer::conv2d<T>(input->out(), {11, 11}, 64, {2, 2}, {4, 4}, {1, 1});
   // TODO batch norm
   // std::cout << "TETETETETETE" << std::endl;

   auto act1 = layer::activation<T>(conv2d1->out(), layer::RELU);
   auto pool1 = layer::pooling<T>(act1->out(), {3, 3}, {1, 1}, {2, 2}, MAX_POOL);

   auto block1 = basic_block(
         pool1->out(), 64, {1, 1});
   auto block2 = basic_block(
         block1.back()->out(), 64, {2, 2});

   auto block3 = basic_block(
         block2.back()->out(), 128, {2, 2});
   auto block4 = basic_block(
         block3.back()->out(), 128, {2, 2});

   auto block5 = basic_block(
         block4.back()->out(), 256, {2, 2});
   auto block6 = basic_block(
         block5.back()->out(), 256, {2, 2});

   auto block7 = basic_block(
         block6.back()->out(), 512, {2, 2});
   auto block8 = basic_block(
         block7.back()->out(), 512, {2, 2});

   auto pool2 = layer::pooling<T>(block8.back()->out(), {2, 2}, {0, 0}, {1, 1}, AVERAGE_POOL);

   auto flatten = layer::flatten<T>(pool2->out());
   auto fc1 = layer::fullyconnected<T>(flatten->out(), train_set.nclasses(), false);
   auto act2 = layer::activation<T>(fc1->out(), layer::SOFTMAX);

   auto output = layer::output<T>(act2->out());
      
   std::vector<layer::Layer<T> *> layers;

   layers.insert(std::end(layers), input);

   layers.insert(std::end(layers), conv2d1);
   layers.insert(std::end(layers), act1);
   layers.insert(std::end(layers), pool1);
   
   layers.insert(std::end(layers), std::begin(block1), std::end(block1));
   layers.insert(std::end(layers), std::begin(block2), std::end(block2));

   layers.insert(std::end(layers), std::begin(block3), std::end(block3));
   layers.insert(std::end(layers), std::begin(block4), std::end(block4));

   layers.insert(std::end(layers), std::begin(block5), std::end(block5));
   layers.insert(std::end(layers), std::begin(block6), std::end(block6));

   layers.insert(std::end(layers), std::begin(block7), std::end(block7));
   layers.insert(std::end(layers), std::begin(block8), std::end(block8));

   layers.insert(std::end(layers), pool2);
   layers.insert(std::end(layers), flatten);
   layers.insert(std::end(layers), fc1);
   layers.insert(std::end(layers), act2);

   layers.insert(std::end(layers), output);

   model::NeuralNetwork<T> model(layers, optimizer::CROSS_ENTROPY, optimizer::SGD, params);

   model::metric_t metrics;

   model.fit(&train_set.images(), &train_set.labels(), metrics, true);

   delete output;

   magmadnn_finalize();

   return 0;

}
