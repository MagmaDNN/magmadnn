/**
 * @file resnet_cifar10.cpp
 * @author Florent Lopez
 * @version 1.0
 * @date 2020-07-21
 *
 * @copyright Copyright (c) 2020
 */

#include "magmadnn.h"

#include <iostream>

using namespace magmadnn;

template <typename T>
std::vector<layer::Layer<T> *> basic_block(
      op::Operation<T>* input, int channels,
      const std::vector<unsigned int> strides) {

   std::cout << "[basic_block]"
             << " input size = " << input->get_output_shape(0)
             << ", " << input->get_output_shape(1)
             << ", " << input->get_output_shape(2)
             << ", " << input->get_output_shape(3)
             << std::endl;

   auto conv2d1 = layer::conv2d<T>(input, {3, 3}, channels, {1, 1}, strides, {1, 1});
   auto bn1 = layer::batchnorm(conv2d1->out());
   auto act1 = layer::activation<T>(bn1->out(), layer::RELU);

   auto conv2d2 = layer::conv2d<T>(act1->out(), {3, 3}, channels, {1, 1}, {1, 1}, {1, 1});
   auto bn2 = layer::batchnorm(conv2d2->out());

   // auto shortcut = op::pow(input, 2, true, false);
   // auto shortcut = op::pow(conv2d2->out(), 1, true, false);
   // auto shortcut = op::add(input, conv2d2->out(), true, false);
   // auto shortcut = layer::shortcut(conv2d2->out(), input);

   std::cout << "[basic_block]"
             << " conv2 output size = " << bn2->out()->get_output_shape(0)
             << ", " << bn2->out()->get_output_shape(1)
             << ", " << bn2->out()->get_output_shape(2)
             << ", " << bn2->out()->get_output_shape(3)
             << std::endl;

   // auto act2 = layer::activation<T>(conv2d2->out(), layer::RELU);
   // auto act2 = layer::activation<T>(shortcut->out(), layer::RELU);
   auto act2 = layer::activation<T>(op::add(bn2->out(), input), layer::RELU);

   std::vector<layer::Layer<T> *> layers =
      {conv2d1, bn1, act1,
       conv2d2, bn2,
       // shortcut,
       act2};

   return layers;
}

int main(int argc, char** argv) {

   std::string context = "resnet_cifar10";
   
   // Data type
   using T = float;

#if defined(MAGMADNN_HAVE_MPI)
   MPI_Init(&argc, &argv);
#endif

   magmadnn_init();

   // Location of the CIFAR-10 dataset
   std::string const cifar10_dir = ".";
   // Load CIFAR-10 trainnig dataset
   magmadnn::data::CIFAR10<T> train_set(cifar10_dir, magmadnn::data::Train);
   magmadnn::data::CIFAR10<T> test_set(cifar10_dir, magmadnn::data::Test);

   // Training parameters
   magmadnn::model::nn_params_t params;
   params.batch_size = 128;
   // params.batch_size = 256;
   params.n_epochs = 500;
   // params.learning_rate = 0.1;
   // params.learning_rate = 0.05;
   // params.learning_rate = 0.01;
   // params.learning_rate = 0.001;
   // params.learning_rate = 0.002;
   params.learning_rate = 1e-4;
   // params.learning_rate = 1e-5;
   // params.learning_rate = 1e-6;
   // params.learning_rate = 1.0;
   // params.decaying_factor = 0.99;
   
   // Memory
   magmadnn::memory_t training_memory_type;
#if defined(MAGMADNN_HAVE_CUDA)
   int devid = 0;
   // cudaSetDevice(1);
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

   std::cout << "[" << context << "]"
             << " input size = " << input->out()->get_output_shape(0)
             << ", " << input->out()->get_output_shape(1)
             << ", " << input->out()->get_output_shape(2)
             << ", " << input->out()->get_output_shape(3)
             << std::endl;

   auto conv2d1 = layer::conv2d<T>(input->out(), {3, 3}, 16, {1, 1}, {1, 1}, {1, 1});
   auto bn1 = layer::batchnorm(conv2d1->out());      
   auto act1 = layer::activation<T>(bn1->out(), layer::RELU);

   std::cout << "[" << context << "]"
             << " input size = " << act1->out()->get_output_shape(0)
             << ", " << act1->out()->get_output_shape(1)
             << ", " << act1->out()->get_output_shape(2)
             << ", " << act1->out()->get_output_shape(3)
             << std::endl;

   // auto act1 = layer::activation<T>(conv2d1->out(), layer::RELU);
   // auto pool1 = layer::pooling<T>(act1->out(), {3, 3}, {1, 1}, {2, 2}, MAX_POOL);
   // auto pool1 = layer::pooling<T>(act1->out(), {3, 3}, {1, 1}, {2, 2}, AVERAGE_POOL);

   // auto block1 = basic_block(
   //       pool1->out(), 64, {1, 1});

   std::cout << "[" << context << "] 16 filters layers" << std::endl;

   auto block1 = basic_block(
         act1->out(), 16, {1, 1});
   auto block2 = basic_block(
         block1.back()->out(), 16, {1, 1});

   std::cout << "[" << context << "] 32 filters layers" << std::endl;

   auto block3 = basic_block(
         block2.back()->out(), 32, {2, 2});
   // auto block4 = basic_block(
   //       block3.back()->out(), 128, {2, 2});

   // auto block5 = basic_block(
   //       block4.back()->out(), 256, {2, 2});
   // auto block6 = basic_block(
   //       block5.back()->out(), 256, {2, 2});

   // auto block7 = basic_block(
   //       block6.back()->out(), 512, {2, 2});
   // auto block8 = basic_block(
   //       block7.back()->out(), 512, {2, 2});

   // auto pool2 = layer::pooling<T>(block1.back()->out(), {2, 2}, {0, 0}, {1, 1}, AVERAGE_POOL);
   // auto pool2 = layer::pooling<T>(block2.back()->out(), {2, 2}, {0, 0}, {1, 1}, AVERAGE_POOL);
   auto pool2 = layer::pooling<T>(block3.back()->out(), {2, 2}, {0, 0}, {1, 1}, AVERAGE_POOL);
   // auto pool2 = layer::pooling<T>(block8.back()->out(), {2, 2}, {0, 0}, {1, 1}, AVERAGE_POOL);
   // auto pool2 = layer::pooling<T>(act1->out(), {2, 2}, {0, 0}, {1, 1}, AVERAGE_POOL);

   auto flatten = layer::flatten<T>(pool2->out());

   auto fc1 = layer::fullyconnected<T>(flatten->out(), train_set.nclasses(), false);
   auto act2 = layer::activation<T>(fc1->out(), layer::SOFTMAX);

   auto output = layer::output<T>(act2->out());
      
   std::vector<layer::Layer<T> *> layers;

   layers.insert(std::end(layers), input);
   
   layers.insert(std::end(layers), conv2d1);
   layers.insert(std::end(layers), bn1);
   layers.insert(std::end(layers), act1);
   // layers.insert(std::end(layers), pool1);

   layers.insert(std::end(layers), std::begin(block1), std::end(block1));
   layers.insert(std::end(layers), std::begin(block2), std::end(block2));

   layers.insert(std::end(layers), std::begin(block3), std::end(block3));
   // layers.insert(std::end(layers), std::begin(block4), std::end(block4));

   // layers.insert(std::end(layers), std::begin(block5), std::end(block5));
   // layers.insert(std::end(layers), std::begin(block6), std::end(block6));

   // layers.insert(std::end(layers), std::begin(block7), std::end(block7));
   // layers.insert(std::end(layers), std::begin(block8), std::end(block8));

   layers.insert(std::end(layers), pool2);
   layers.insert(std::end(layers), flatten);
   layers.insert(std::end(layers), fc1);
   layers.insert(std::end(layers), act2);

   layers.insert(std::end(layers), output);

   model::NeuralNetwork<T> model(layers, optimizer::CROSS_ENTROPY, optimizer::SGD, params);

   model::metric_t metrics;

   // std::cout << "TETETETETETE" << std::endl;

   model.fit(&train_set.images(), &train_set.labels(), metrics, true);

   delete output;

   magmadnn_finalize();

#if defined(MAGMADNN_HAVE_MPI)
   MPI_Finalize();
#endif

   return 0;

}
