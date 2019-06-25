/**
 * @file simple_network.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-30
 * 
 * @copyright Copyright (c) 2019
 */
#include <cstdio>
#include <vector>
#include <cstdint>

/* we must include magmadnn as always */
#include "magmadnn.h"

/* tell the compiler we're using functions from the magmadnn namespace */
using namespace magmadnn;

/* these are used for reading in the MNIST data set -- found at http://yann.lecun.com/exdb/mnist/ */
Tensor<float> *read_mnist_images(const char *file_name, uint32_t &n_images, uint32_t &n_rows, uint32_t &n_cols);
Tensor<float> *read_mnist_labels(const char *file_name, uint32_t &n_labels, uint32_t n_classes);
void print_image(uint32_t image_idx, Tensor<float> *images, Tensor<float> *labels, uint32_t n_rows, uint32_t n_cols);



int main(int argc, char **argv) {
    /* every magmadnn program must begin with magmadnn_init. This allows magmadnn to test the environment
        and initialize some GPU data. */
    magmadnn_init();

    /* here we declare our variables for the simulation */
    Tensor<float> *images_host, *labels_host;
    uint32_t n_images, n_rows, n_cols, n_labels, n_classes = 10, n_features;

    /* these functions read-in and return tensors holding the mnist data set
        to use them, please change the string to the path to your local copy of the mnist dataset.
        it can be downloaded from http://yann.lecun.com/exdb/mnist */
    images_host = read_mnist_images("/opt/data/mnist/train-images-idx3-ubyte", n_images, n_rows, n_cols);
    labels_host = read_mnist_labels("/opt/data/mnist/train-labels-idx1-ubyte", n_labels, n_classes);

    n_features = n_rows * n_cols;

    /* exit on error */
    if (images_host == NULL || labels_host == NULL) {
        return 1;
    }

    /* if you run this program with a number argument, it will print out that number sample on the command line */
    if (argc == 2) {
        print_image(std::atoi(argv[1]), images_host, labels_host, n_rows, n_cols);
    }

    /* initialize our model parameters */
    model::nn_params_t params;
    params.batch_size = 32;    /* batch size: the number of samples to process in each mini-batch */
    params.n_epochs = 5;    /* # of epochs: the number of passes over the entire training set */

    /* INITIALIZING THE NETWORK */
    /* create a variable (of type float) with size  (batch_size x n_features) 
        This will serve as the input to our network. */
    auto x_batch = op::var<float>("x_batch", {params.batch_size, n_features}, {NONE,{}}, DEVICE);

    /* initialize the layers in our network */
    auto input = layer::input(x_batch);
    auto fc1 = layer::fullyconnected(input->out(), 512);
    auto act1 = layer::activation(fc1->out(), layer::RELU);

    auto fc2 = layer::fullyconnected(act1->out(), 256);
    auto act2 = layer::activation(fc2->out(), layer::RELU);

    auto fc3 = layer::fullyconnected(act2->out(), n_classes);
    auto act3 = layer::activation(fc3->out(), layer::SOFTMAX);

    auto output = layer::output(act3->out());

    /* wrap each layer in a vector of layers to pass to the model */
    std::vector<layer::Layer<float> *> layers = {input, fc1, act1, fc2, act2, fc3, act3, output};

    /* this creates a Model for us. The model can train on our data and perform other typical operations
        that a ML model can. 
        layers: the previously created vector of layers containing our network
        loss_func: use cross entropy as our loss function
        optimizer: use stochastic gradient descent to optimize our network
        params: the parameter struct created earlier with our network settings */
    model::NeuralNetwork<float> model(layers, optimizer::CROSS_ENTROPY, optimizer::SGD, params);

    /* metric_t records the model metrics such as accuracy, loss, and training time */
    model::metric_t metrics;

    /* fit will train our network using the given settings.
        X: independent data
        y: ground truth
        metrics: metric struct to store run time metrics
        verbose: whether to print training info during training or not */
    model.fit(images_host, labels_host, metrics, true);

    /* clean up memory after training */
    delete images_host;
    delete labels_host;
    delete output;

    /* every magmadnn program should call magmadnn_finalize before exiting */
    magmadnn_finalize();

    return 0;
}




#define FREAD_CHECK(res, nmemb) if((res) != (nmemb)) { fprintf(stderr, "fread fail.\n"); return NULL; }

inline void endian_swap(uint32_t &val) {
    /* taken from https://stackoverflow.com/questions/13001183/how-to-read-little-endian-integers-from-file-in-c */
    val = (val >> 24) | ((val << 8) & 0xff0000) | ((val >> 8) & 0xff00) | (val << 24);
}

Tensor<float> *read_mnist_images(const char *file_name, uint32_t &n_images, uint32_t &n_rows, uint32_t &n_cols) {
    FILE *file;
    unsigned char magic[4];
    Tensor<float> *data;
    uint8_t val;

    file = std::fopen(file_name, "r");
    
    if (file == NULL) {
        std::fprintf(stderr, "Could not open %s for reading.\n", file_name);
        return NULL;
    }

    FREAD_CHECK(fread(magic, sizeof(char), 4, file), 4);
    if (magic[2] != 0x08 || magic[3] != 0x03) {
        std::fprintf(stderr, "Bad file magic.\n");
        return NULL;
    }

    FREAD_CHECK(fread(&n_images, sizeof(uint32_t), 1, file), 1);
    endian_swap(n_images);

    FREAD_CHECK(fread(&n_rows, sizeof(uint32_t), 1, file), 1);
    endian_swap(n_rows);

    FREAD_CHECK(fread(&n_cols, sizeof(uint32_t), 1, file), 1);
    endian_swap(n_cols);

    printf("Preparing to read %u images with size %u x %u ...\n", n_images, n_rows, n_cols);

    char bytes[n_rows * n_cols];

    /* allocate tensor */
    data = new Tensor<float> ({n_images, n_rows, n_cols}, {NONE,{}}, HOST);

    for (uint32_t i = 0; i < n_images; i++) {
        FREAD_CHECK(fread(bytes, sizeof(char), n_rows * n_cols, file), n_rows*n_cols);

        for (uint32_t r = 0; r < n_rows; r++) {
            for (uint32_t c = 0; c < n_cols; c++) {
                val = bytes[r*n_cols + c];

                data->set(i*n_rows*n_cols + r*n_cols + c, (val/128.0f) - 1.0f);
            }
        }
    }
    printf("finished reading images.\n");

    fclose(file);

    return data;
}

Tensor<float> *read_mnist_labels(const char *file_name, uint32_t &n_labels, uint32_t n_classes) {
    FILE *file;
    unsigned char magic[4];
    Tensor<float> *labels;
    uint8_t val;

    file = std::fopen(file_name, "r");
    
    if (file == NULL) {
        std::fprintf(stderr, "Could not open %s for reading.\n", file_name);
        return NULL;
    }

    FREAD_CHECK(fread(magic, sizeof(char), 4, file), 4);

    if (magic[2] != 0x08 || magic[3] != 0x01) {
        std::fprintf(stderr, "Bad file magic.\n");
        return NULL;
    }

    FREAD_CHECK(fread(&n_labels, sizeof(uint32_t), 1, file), 1);
    endian_swap(n_labels);

    printf("Preparing to read %u labels with %u classes ...\n", n_labels, n_classes);


    /* allocate tensor */
    labels = new Tensor<float> ({n_labels, n_classes}, {ZERO,{}}, HOST);

    printf("finished reading labels.\n");

    for (unsigned int i = 0; i < n_labels; i++) {
        FREAD_CHECK(fread(&val, sizeof(char), 1, file), 1);

        labels->set(i*n_classes + val, 1.0f);
    }

    fclose(file);


    return labels;
}

void print_image(uint32_t image_idx, Tensor<float> *images, Tensor<float> *labels, uint32_t n_rows, uint32_t n_cols) {
    uint8_t label = 0;
    uint32_t n_classes = labels->get_shape(1);

    /* assign the label */
    for (uint32_t i = 0; i < n_classes; i++) {

        if (std::fabs(labels->get(image_idx*n_classes + i) - 1.0f) <= 1E-8) {
            label = i;
            break;
        }
    }

    printf("Image[%u] is digit %u:\n", image_idx, label);

    for (unsigned int r = 0; r < n_rows; r++) {
        for (unsigned int c = 0; c < n_cols; c++) {
            printf("%03u ", (uint8_t) ((images->get(image_idx*n_rows*n_cols + r*n_cols + c)+1.0f)*128.0f) );
        }
        printf("\n");
    }
}

#undef FREAD_CHECK