/**
 * @file resnet_cifar10.cpp
 * @author Florent Lopez
 * @version 1.0
 * @date 2020-07-21
 *
 * @copyright Copyright (c) 2020
 */

#include "Arguments.h"
#include "magmadnn.h"
#include "models/resnet.h"

#include <iostream>

using namespace magmadnn;

int main(int argc, char** argv) {

   std::string context = "resnet_cifar10";
   
   // Data type
   using T = float;
   
#if defined(MAGMADNN_HAVE_MPI)
   MPI_Init(&argc, &argv);
#endif

   magmadnn_init();

   magmadnn::Arguments args;
   args.parse(context, argc, argv);

   // Location of the CIFAR-10 dataset
   std::string const cifar10_dir = ".";
   // Load CIFAR-10 trainnig dataset
   magmadnn::data::CIFAR10<T> train_set(cifar10_dir, magmadnn::data::Train);
   magmadnn::data::CIFAR10<T> test_set(cifar10_dir, magmadnn::data::Test);

   // Training parameters
   magmadnn::model::nn_params_t params;
   params.batch_size = 128;
   // params.batch_size = 256;
   // params.n_epochs = 500;
   params.n_epochs = 50;

   if (args.learning_rate > 0) {
      params.learning_rate = args.learning_rate; 
   }
   else {
      // params.learning_rate = 0.1;
      // params.learning_rate = 0.05;
      // params.learning_rate = 0.01;
      params.learning_rate = 0.001;
      // params.learning_rate = 0.002;
      // params.learning_rate = 1e-4;
      // params.learning_rate = 1e-5;
      // params.learning_rate = 1e-6;
      // params.learning_rate = 1.0;
      // params.decaying_factor = 0.99;
   }

   // Number of stacked blocks per filter sizes
   // int num_stacked_blocks = 1;
   int num_stacked_blocks = 2;
   
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

   // bool enable_shortcut = true;
   bool enable_shortcut = args.enable_shortcut;
   
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

   // std::cout << "[" << context << "]"
   //           << " input size = " << input->out()->get_output_shape(0)
   //           << ", " << input->out()->get_output_shape(1)
   //           << ", " << input->out()->get_output_shape(2)
   //           << ", " << input->out()->get_output_shape(3)
   //           << std::endl;4

   auto conv2d1 = layer::conv2d<T>(input->out(), {3, 3}, 16, {1, 1}, {1, 1}, {1, 1});
   auto bn1 = layer::batchnorm(conv2d1->out());      
   auto act1 = layer::activation<T>(bn1->out(), layer::RELU);

   // std::cout << "[" << context << "]"
   //           << " input size = " << act1->out()->get_output_shape(0)
   //           << ", " << act1->out()->get_output_shape(1)
   //           << ", " << act1->out()->get_output_shape(2)
   //           << ", " << act1->out()->get_output_shape(3)
   //           << std::endl;

   // auto act1 = layer::activation<T>(conv2d1->out(), layer::RELU);
   // auto pool1 = layer::pooling<T>(act1->out(), {3, 3}, {1, 1}, {2, 2}, MAX_POOL);
   // auto pool1 = layer::pooling<T>(act1->out(), {3, 3}, {1, 1}, {2, 2}, AVERAGE_POOL);

   // auto block1 = basic_block(
   //       pool1->out(), 64, {1, 1});

   // std::cout << "[" << context << "] 16 filters layers" << std::endl;

   std::vector<layer::Layer<T> *> blocks(0, nullptr);

   for (int i = 0; i < num_stacked_blocks; ++i) {

      op::Operation<T>* block1_input = nullptr;
      
      if (i == 0) {
         // First block
         block1_input = act1->out();
      }
      else {
         // Subsequent block: input from previous stacked block output
         block1_input = blocks.back()->out();
      }
         
      auto block1 = basic_block(
            block1_input, 16, {1, 1}, enable_shortcut);
      auto block2 = basic_block(
            block1.back()->out(), 16, {1, 1}, enable_shortcut);

      blocks.insert(std::end(blocks), std::begin(block1), std::end(block1));
      blocks.insert(std::end(blocks), std::begin(block2), std::end(block2));      
   }
   
   // std::cout << "[" << context << "] 32 filters layers" << std::endl;

   for (int i = 0; i < num_stacked_blocks; ++i) {

      std::vector<unsigned int> strides = {1, 1};
      bool enable_shortcut_input = enable_shortcut; 
      
      if (i == 0) {
         // Downsampling
         strides = {2, 2};
         // enable_shortcut_input = false;
      }
      
      auto block3 = basic_block(
            blocks.back()->out(), 32, strides, enable_shortcut_input);
      auto block4 = basic_block(
            block3.back()->out(), 32, {1, 1}, enable_shortcut);

      blocks.insert(std::end(blocks), std::begin(block3), std::end(block3));
      blocks.insert(std::end(blocks), std::begin(block4), std::end(block4));
   }
   
   // std::cout << "[" << context << "] 64 filters layers" << std::endl;

   for (int i = 0; i < num_stacked_blocks; ++i) {

      std::vector<unsigned int> strides = {1, 1};
      bool enable_shortcut_input = enable_shortcut; 

      if (i == 0) {
         // Downsampling
         strides = {2, 2};
         // enable_shortcut_input = false;
      }

      auto block5 = basic_block(
            blocks.back()->out(), 64, strides, enable_shortcut_input);
      auto block6 = basic_block(
            block5.back()->out(), 64, {1, 1}, enable_shortcut);

      blocks.insert(std::end(blocks), std::begin(block5), std::end(block5));
      blocks.insert(std::end(blocks), std::begin(block6), std::end(block6));
   }
      
   // auto pool2 = layer::pooling<T>(block6.back()->out(), {2, 2}, {0, 0}, {1, 1}, AVERAGE_POOL);
   // auto pool2 = layer::pooling<T>(act1->out(), {2, 2}, {0, 0}, {1, 1}, AVERAGE_POOL);
   auto pool2 = layer::pooling<T>(blocks.back()->out(), {2, 2}, {0, 0}, {1, 1}, AVERAGE_POOL);
   
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

   layers.insert(std::end(layers), std::begin(blocks), std::end(blocks));

   // layers.insert(std::end(layers), std::begin(block1), std::end(block1));
   // layers.insert(std::end(layers), std::begin(block1), std::end(block1));

   // layers.insert(std::end(layers), std::begin(block3), std::end(block3));
   // layers.insert(std::end(layers), std::begin(block4), std::end(block4));

   // layers.insert(std::end(layers), std::begin(block5), std::end(block5));
   // layers.insert(std::end(layers), std::begin(block6), std::end(block6));

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
