## Tutorial 05: First Neural Network
------------------------------------

Writing training routines for neural networks can be a difficult task for beginners, however, MagmaDNN provides the `Model` class, which only requires the coder to define the network and it will take care of training. The model we'll be using is `magmadnn::model::NeuralNetwork<T>`. 


For a full neural network example using the MNIST data set see [the simple_network example](/docs/simple_network.cpp).


Neural Network models take four parameters on creation: a vector of layers, loss function, optimizer, and parameter struct. For example

```c++
/* initialize our model parameters */
model::nn_params_t params;
params.batch_size = 100;    /* batch size: the number of samples to process in each mini-batch */
params.n_epochs = 5;    /* # of epochs: the number of passes over the entire training set */
params.learning_rate = 0.05;    /* learning rate of model */

auto input = layer::input(x_batch);
auto fc = layer::input(input->out(), 10);
auto act = layer::activation(fc->out(), layer::SOFTMAX);
auto output = layer::output(act->out());
std::vector<layer::Layer<float> *> layers = {input, fc, act, output};   /* create our vector of layers */

/* use cross entropy loss and stochastic gradient descent as an optimizer */
model::NeuralNetwork<float> model(layers, optimizer::CROSS_ENTROPY, optimizer::SGD, params);
```


And that is it! Our model has been made and is ready for training. All `Model<T>` subclasses define a `fit` function, which will train on the data set. The fit function takes the data set, labels, metric struct, and verbosity options. For instance

```c++
model::metric_t metric_out;
bool is_verbose = true;

/* x_train and y_train are tensors containing the data set */
model.fit(x_train, y_train, labels, metric_out, is_verbose);

/* metric will now store the training accuracy, loss, and duration */
```
