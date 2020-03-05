/**
 * @file alexnet.cpp
 * @author Florent Lopez
 * @version 1.0
 * @date 2020-03-04
 *
 * @copyright Copyright (c) 2020
 */

#include "magmadnn.h"

#include <iostream>

using namespace magmadnn;

int main(int argc, char** argv) {

   // Data type
   using T = float;

   magmadnn_init();

   // Location of the CIFAR-10 dataset
   std::string const cifar10_dir = ".";
   // Location of the CIFAR-100 dataset
   std::string const cifar100_dir = ".";
   // Load CIFAR trainnig dataset
   // magmadnn::data::CIFAR10<T> train_set(cifar10_dir, magmadnn::data::Train);
   // magmadnn::data::CIFAR10<T> test_set(cifar10_dir, magmadnn::data::Test);

   magmadnn::data::CIFAR100<T> train_set(cifar100_dir, magmadnn::data::Train);
   magmadnn::data::CIFAR100<T> test_set(cifar100_dir, magmadnn::data::Test);

   // Training parameters
   magmadnn::model::nn_params_t params;
   params.batch_size = 128;
   params.n_epochs = 500;
   // params.learning_rate = 0.05;
   params.learning_rate = 0.001;
   // params.learning_rate = 0.0001;
   // params.learning_rate = 1.0;
   // params.decaying_factor = 0.99;
   
   // Memory
   magmadnn::memory_t training_memory_type;
#if defined(MAGMADNN_HAVE_CUDA)
   training_memory_type = DEVICE;
#else
   training_memory_type = HOST;
#endif

   std::cout << "[alexnet] Image dimensions: " << train_set.nrows() << " x " << train_set.ncols() << std::endl;
   std::cout << "[alexnet] Number of chanels: " << train_set.nchanels() << std::endl;
   std::cout << "[alexnet] Number of classes: " << train_set.nclasses() << std::endl;
   std::cout << "[alexnet] Training set size: " << train_set.nimages() << std::endl;
   
   auto x_batch = op::var<T>(
         "x_batch",
         {params.batch_size, train_set.nchanels(),  train_set.nrows(), train_set.ncols()},
         {NONE, {}},
         training_memory_type);

   auto input = layer::input<T>(x_batch);

   auto conv2d1 = layer::conv2d<T>(input->out(), {11, 11}, 64, {2, 2}, {4, 4}, {1, 1});
   // auto conv2d1 = layer::conv2d<T>(input->out(), {11, 11}, 64, layer::SAME, {4, 4}, {1, 1});
   auto act1 = layer::activation<T>(conv2d1->out(), layer::RELU);
   // auto pool1 = layer::pooling<T>(act1->out(), {3, 3}, layer::SAME, {2, 2}, AVERAGE_POOL);
   auto pool1 = layer::pooling<T>(act1->out(), {3, 3}, {0, 0}, {2, 2}, AVERAGE_POOL);
   // auto pool1 = layer::pooling<T>(act1->out(), {3, 3}, {0, 0}, {2, 2}, MAX_POOL);
   // auto pool1 = layer::pooling<T>(act1->out(), {3, 3}, layer::SAME, {1, 1}, MAX_POOL);
   // auto pool1 = layer::pooling<T>(act1->out(), {3, 3}, layer::VALID, {2, 2}, MAX_POOL);

   // auto conv2d2 = layer::conv2d<T>(pool1->out(), {5, 5}, 192, layer::VALID, {1, 1}, {1, 1});
   auto conv2d2 = layer::conv2d<T>(pool1->out(), {5, 5}, 192, {2, 2}, {1, 1}, {1, 1});
   auto act2 = layer::activation<T>(conv2d2->out(), layer::RELU);
   auto pool2 = layer::pooling<T>(act2->out(), {3, 3}, {0, 0}, {2, 2}, AVERAGE_POOL);
   // auto pool2 = layer::pooling<T>(act2->out(), {3, 3}, {0, 0}, {2, 2}, MAX_POOL);
   // auto pool2 = layer::pooling<T>(act2->out(), {3, 3}, layer::SAME, {2, 2}, MAX_POOL);

   auto conv2d3 = layer::conv2d<T>(pool2->out(), {3, 3}, 384, {1, 1}, {1, 1}, {1, 1});
   // auto conv2d3 = layer::conv2d<T>(act2->out(), {3, 3}, 256, {1, 1}, {1, 1}, {1, 1});
   auto act3 = layer::activation<T>(conv2d3->out(), layer::RELU);

   auto conv2d4 = layer::conv2d<T>(act3->out(), {3, 3}, 384, {1, 1}, {1, 1}, {1, 1});
   auto act4 = layer::activation<T>(conv2d4->out(), layer::RELU);

   auto conv2d5 = layer::conv2d<T>(act4->out(), {3, 3}, 256, {1, 1}, {1, 1}, {1, 1});
   auto act5 = layer::activation<T>(conv2d5->out(), layer::RELU);

   // auto pool3 = layer::pooling<T>(act3->out(), {3, 3}, {0, 0}, {2, 2}, AVERAGE_POOL);
   auto pool3 = layer::pooling<T>(act5->out(), {3, 3}, layer::SAME, {2, 2}, AVERAGE_POOL);
   // auto pool3 = layer::pooling<T>(act5->out(), {3, 3}, {0, 0}, {2, 2}, MAX_POOL);
   // auto pool3 = layer::pooling<T>(act5->out(), {3, 3}, layer::SAME, {2, 2}, MAX_POOL);

   auto dropout1 = layer::dropout<float>(pool3->out(), 0.5);

   // auto flatten = layer::flatten<T>(pool3->out());
   auto flatten = layer::flatten<T>(dropout1->out());

   auto fc1 = layer::fullyconnected<T>(flatten->out(), 4096, true);
   auto act6 = layer::activation<T>(fc1->out(), layer::RELU);

   auto fc2 = layer::fullyconnected<T>(act6->out(), 4096, true);
   auto act7 = layer::activation<T>(fc2->out(), layer::RELU);

   auto fc3 = layer::fullyconnected<T>(act7->out(), train_set.nclasses(), false);
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

   model::NeuralNetwork<float> model(layers, optimizer::CROSS_ENTROPY, optimizer::SGD, params);
   // model::NeuralNetwork<float> model(layers, optimizer::CROSS_ENTROPY, optimizer::RMSPROP, params);
   // model::NeuralNetwork<float> model(layers, optimizer::MSE, optimizer::SGD, params);

   model::metric_t metrics;

   model.fit(&train_set.images(), &train_set.labels(), metrics, true);

   delete output;

   magmadnn_finalize();

   return 0;

}
