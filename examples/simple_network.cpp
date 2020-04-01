/**
 * @file simple_network.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-30
 *
 * @copyright Copyright (c) 2019
 */
#include <cstdint>
#include <cstdio>
#include <vector>
#include <iostream>

/* we must include magmadnn as always */
#include "magmadnn.h"

/* tell the compiler we're using functions from the magmadnn namespace */
using namespace magmadnn;

// void print_image(uint32_t image_idx, Tensor<float> *images, Tensor<float> *labels, uint32_t n_rows, uint32_t n_cols);

int main(int argc, char **argv) {

   using T = float;
   
   // Every magmadnn program must begin with magmadnn_init. This
   // allows magmadnn to test the environment and initialize some GPU
   // data.
    magmadnn_init();

    // Location of the MNIST dataset
    std::string const mnist_dir = ".";
    // Load MNIST trainnig dataset
    magmadnn::data::MNIST<T> train_set(mnist_dir, magmadnn::data::Train);
    magmadnn::data::MNIST<T> test_set(mnist_dir, magmadnn::data::Test);

    // Number of features
    uint32_t n_features;
    // Memory used for training (CPU or GPU)
    memory_t training_memory_type;

    n_features = train_set.nrows() * train_set.ncols();

    /* if you run this program with a number argument, it will print out that number sample on the command line */
    if (argc == 2) {
       train_set.print_image(
             std::atoi(argv[1]));
    }

    // Initialize our model parameters
    model::nn_params_t params;
    params.batch_size = 32; /* batch size: the number of samples to process in each mini-batch */
    params.n_epochs = 5;    /* # of epochs: the number of passes over the entire training set */
    params.learning_rate = 0.05;

    // This is only necessary for a general example which can handle
    // all install types. Typically, When you write your own MagmaDNN
    // code you will not need macros as you can simply pass DEVICE to
    // the x_batch constructor.
#if defined(MAGMADNN_HAVE_CUDA)
    training_memory_type = DEVICE;
    std::cout << "Training on GPUs" << std::endl;
#else
    training_memory_type = HOST;
    std::cout << "Training on CPUs" << std::endl;
#endif

    // Creating the network

    // Create a variable (of type T=float) with size (batch_size x
    // n_features) This will serve as the input to our network.
    auto x_batch = op::var<T>("x_batch", {params.batch_size, n_features}, {NONE, {}}, training_memory_type);

    // Initialize the layers in our network
    auto input = layer::input(x_batch);
    auto fc1 = layer::fullyconnected(input->out(), 784, false);
    auto act1 = layer::activation(fc1->out(), layer::RELU);

    auto fc2 = layer::fullyconnected(act1->out(), 500, false);
    auto act2 = layer::activation(fc2->out(), layer::RELU);

    auto fc3 = layer::fullyconnected(act2->out(), train_set.nclasses(), false);
    auto act3 = layer::activation(fc3->out(), layer::SOFTMAX);

    auto output = layer::output(act3->out());

    // Wrap each layer in a vector of layers to pass to the model
    std::vector<layer::Layer<float> *> layers = {input, fc1, act1, fc2, act2, fc3, act3, output};

    // This creates a Model for us. The model can train on our data
    // and perform other typical operations that a ML model can.
    //                                    
    // - layers: the previously created vector of layers containing our
    // network
    // - loss_func: use cross entropy as our loss function
    // - optimizer: use stochastic gradient descent to optimize our
    // network
    // - params: the parameter struct created earlier with our network
    // settings
    model::NeuralNetwork<float> model(layers, optimizer::CROSS_ENTROPY, optimizer::SGD, params);

    // metric_t records the model metrics such as accuracy, loss, and
    // training time
    model::metric_t metrics;

    /* fit will train our network using the given settings.
        X: independent data
        y: ground truth
        metrics: metric struct to store run time metrics
        verbose: whether to print training info during training or not */
    model.fit(&train_set.images(), &train_set.labels(), metrics, true);

    // Clean up memory after training
    delete output;

    // Every magmadnn program should call magmadnn_finalize before
    // exiting
    magmadnn_finalize();

    return 0;
}
