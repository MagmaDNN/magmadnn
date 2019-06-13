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
#include "magmadnn.h"

using namespace magmadnn;

Tensor<float> *read_mnist_images(const char *file_name, uint32_t &n_images, uint32_t &n_rows, uint32_t &n_cols);
Tensor<float> *read_mnist_labels(const char *file_name, uint32_t &n_labels, uint32_t n_classes);
void print_image(uint32_t image_idx, Tensor<float> *images, Tensor<float> *labels);

int main(int argc, char **argv) {
    magmadnn_init();

    Tensor<float> *images_host, *images, *labels_host, *labels;
    uint32_t n_images, n_rows, n_cols, n_labels, n_classes = 10, n_features;

    images_host = read_mnist_images("~/data/mnist/train-images-idx3-ubyte", n_images, n_rows, n_cols);
    labels_host = read_mnist_labels("~/data/mnist/train-labels-idx1-ubyte", n_labels, n_classes);

    n_features = n_rows * n_cols;

    if (images_host == NULL) {
        return 1;
    } else {
        images = new Tensor<float> ({(unsigned int)n_images, (unsigned int)n_features}, {NONE, {}}, DEVICE);
        images->copy_from(*images_host);
    }

    if (labels_host == NULL) {
        return 1;
    } else {
        labels = new Tensor<float> (labels_host->get_shape(), {NONE, {}}, DEVICE);
        labels->copy_from(*labels_host);
    }

    if (argc == 2) {
        print_image(std::atoi(argv[1]), images, labels);
    }

    model::nn_params_t params;
    params.batch_size = 100;
    params.n_epochs = 2;

    auto x_batch = op::var<float>("x_batch", {params.batch_size, n_features}, {NONE,{}}, DEVICE);

    auto input = layer::input(x_batch);
    auto fc1 = layer::fullyconnected(input->out(), 728);
    auto act1 = layer::activation(fc1->out(), layer::SIGMOID);
    auto fc2 = layer::fullyconnected(act1->out(), 256);
    auto act2 = layer::activation(fc2->out(), layer::SIGMOID);
    auto fc3 = layer::fullyconnected(act2->out(), n_classes);
    auto act3 = layer::activation(fc3->out(), layer::SIGMOID);
    auto output = layer::output(act3->out());

    std::vector<layer::Layer<float> *> layers = {input, fc1, act1, fc2, act2, fc3, act3, output};

    model::NeuralNetwork<float> model(layers, optimizer::CROSS_ENTROPY, optimizer::SGD, params);

    model::metric_t metrics;
    model.fit(images, labels, metrics, true);

    delete images_host;
    delete images;
    delete labels_host;
    delete labels;

    magmadnn_finalize();

    return 0;
}

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

    fread(magic, sizeof(char), 4, file);

    if (magic[2] != 0x08 || magic[3] != 0x03) {
        std::fprintf(stderr, "Bad file magic.\n");
        return NULL;
    }

    fread(&n_images, sizeof(uint32_t), 1, file);
    endian_swap(n_images);

    fread(&n_rows, sizeof(uint32_t), 1, file);
    endian_swap(n_rows);

    fread(&n_cols, sizeof(uint32_t), 1, file);
    endian_swap(n_cols);

    printf("Preparing to read %u images with size %u x %u ...\n", n_images, n_rows, n_cols);

    char bytes[n_rows * n_cols];

    /* allocate tensor */
    data = new Tensor<float> ({n_images, n_rows, n_cols}, {NONE,{}}, HOST);

    for (uint32_t i = 0; i < n_images; i++) {
        fread(bytes, sizeof(char), n_rows * n_cols, file);

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

    fread(magic, sizeof(char), 4, file);

    if (magic[2] != 0x08 || magic[3] != 0x01) {
        std::fprintf(stderr, "Bad file magic.\n");
        return NULL;
    }

    fread(&n_labels, sizeof(uint32_t), 1, file);
    endian_swap(n_labels);

    printf("Preparing to read %u labels with %u classes ...\n", n_labels, n_classes);


    /* allocate tensor */
    labels = new Tensor<float> ({n_labels, n_classes}, {ZERO,{}}, HOST);

    printf("finished reading labels.\n");

    for (unsigned int i = 0; i < n_labels; i++) {
        fread(&val, sizeof(char), 1, file);

        labels->set(i*n_classes + val, 1.0f);
    }

    fclose(file);


    return labels;
}

void print_image(uint32_t image_idx, Tensor<float> *images, Tensor<float> *labels) {
    uint8_t label = 0;
    uint32_t n_classes = labels->get_shape(1);
    uint32_t n_rows = images->get_shape(1);
    uint32_t n_cols = images->get_shape(2);

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