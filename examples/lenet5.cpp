/**
 * @file lenet5.cpp
 * @author Florent Lopez
 * @version 1.0
 * @date 2020-03-03
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

   // Location of the MNIST dataset
   std::string const mnist_dir = ".";
   // Load MNIST trainnig dataset
   magmadnn::data::MNIST<T> train_set(mnist_dir, magmadnn::data::Train);
   magmadnn::data::MNIST<T> test_set(mnist_dir, magmadnn::data::Test);

   // Training parameters
   magmadnn::model::nn_params_t params;
   params.batch_size = 128;
   params.n_epochs = 20;
   // params.learning_rate = 0.05;
   params.learning_rate = 0.1;

   // Memory 
   magmadnn::memory_t training_memory_type;
#if defined(MAGMADNN_HAVE_CUDA)
   training_memory_type = DEVICE;
#else
   training_memory_type = HOST;
#endif

   auto x_batch = op::var<T>(
         "x_batch",
         {params.batch_size, train_set.nchanels(),  train_set.nrows(), train_set.ncols()},
         {NONE, {}},
         training_memory_type);

   auto input = layer::input<T>(x_batch);
   
   auto conv2d1 = layer::conv2d<T>(input->out(), {5, 5}, 32, {0, 0}, {1, 1}, {1, 1});
   auto act1 = layer::activation<T>(conv2d1->out(), layer::TANH);
   auto pool1 = layer::pooling<T>(act1->out(), {2, 2}, {0, 0}, {2, 2}, AVERAGE_POOL);

   auto conv2d2 = layer::conv2d<T>(pool1->out(), {5, 5}, 32, {0, 0}, {1, 1}, {1, 1});
   auto act2 = layer::activation<T>(conv2d2->out(), layer::TANH);
   auto pool2 = layer::pooling<T>(act2->out(), {2, 2}, {0, 0}, {2, 2}, AVERAGE_POOL);

   auto flatten = layer::flatten<T>(pool2->out());

   auto fc1 = layer::fullyconnected<T>(flatten->out(), 120, true);
   auto act3 = layer::activation<T>(fc1->out(), layer::TANH);

   auto fc2 = layer::fullyconnected<T>(act3->out(), 84, true);
   auto act4 = layer::activation<T>(fc2->out(), layer::TANH);

   auto fc3 = layer::fullyconnected<T>(act4->out(), train_set.nclasses(), false);
   auto act5 = layer::activation<T>(fc3->out(), layer::SOFTMAX);

   auto output = layer::output<T>(act5->out());

   std::vector<layer::Layer<T> *> layers =
      {input,
       conv2d1, act1, pool1,
       conv2d2, act2, pool2,
       flatten,
       fc1, act3,
       fc2, act4,
       fc3, act5,
       output};

   model::NeuralNetwork<T> model(layers, optimizer::CROSS_ENTROPY, optimizer::SGD, params);
   // model::NeuralNetwork<T> model(layers, optimizer::MSE, optimizer::SGD, params);

   model::metric_t metrics;
   model.fit(&train_set.images(), &train_set.labels(), metrics, true);

   // Compute accuracy of the model on the test set
  
   uint32_t total_correct = 0;
   
   Tensor<T> sample({train_set.nchanels(), test_set.nrows(), test_set.ncols()}, {NONE, {}}, test_set.images().get_memory_type());

   for (uint32_t i = 0; i < test_set.images().get_shape(0); ++i) {

      sample.copy_from(test_set.images(), i * sample.get_size(), sample.get_size());

      auto predicted_class = model.predict_class(&sample);

      auto actual_class = test_set.nclasses() + 1;
      for (uint32_t j = 0; j < test_set.nclasses(); j++) {
         if (std::fabs(test_set.labels().get(i * test_set.nclasses() + j) - 1.0f) <= 1E-8) {
            actual_class = j;
            break;
         }
      }

      if (actual_class == predicted_class) {
         total_correct++;
      } 
   }

   double accuracy = static_cast<double>(total_correct) / static_cast<double>(test_set.images().get_shape(0));
   std::cout << "Model accuracy on testset: " << accuracy << std::endl;
   
   delete output;

   magmadnn_finalize();

   return 0;
}
