/**
 * @file mnist_interactive.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-06-25
 * 
 * @copyright Copyright (c) 2019
 */
#include <cstdio>
#include <vector>
#include <cstdint>
#include <iostream>
#include <iomanip>

#include "magmadnn.h"

using namespace magmadnn;

/* these are used for reading in the MNIST data set -- found at http://yann.lecun.com/exdb/mnist/ */
Tensor<float> *read_mnist_images(const char *file_name, uint32_t &n_images, uint32_t &n_rows, uint32_t &n_cols);
Tensor<float> *read_mnist_labels(const char *file_name, uint32_t &n_labels, uint32_t n_classes);
void print_image(uint32_t image_idx, Tensor<float> *images, Tensor<float> *labels, uint32_t n_rows, uint32_t n_cols);

void show_sample(Tensor<float> *images, Tensor<float> *labels, uint32_t n_rows, uint32_t n_cols);
void train(model::NeuralNetwork<float> &model, Tensor<float> *images, Tensor<float> *labels);
void predict(model::NeuralNetwork<float> &model, Tensor<float> *images, Tensor<float> *labels, uint32_t n_rows, uint32_t n_cols);


int main(int argc, char **argv) {
    magmadnn_init();

    std::string images_path, labels_path;
    Tensor<float> *images_host, *labels_host;
    uint32_t n_images, n_rows, n_cols, n_labels, n_classes = 10, n_features;
    bool use_gpu = true;
    int64_t input;

    /* process args */
    if (argc != 3 && argc != 4) {
        std::cerr << "usage: " << argv[0] << " <path-to-mnist-training-data> <path-to-mnist-training-labels> <use_gpu(optional):y or n>" << std::endl;
        exit(1);
    } else {
        images_path = std::string(argv[1]);
        labels_path = std::string(argv[2]);

        if (argc == 4) {
            use_gpu = (argv[3][0] == 'Y' || argv[3][0] == 'y');
        }
    }

    
    /* dataset can be downloaded from http://yann.lecun.com/exdb/mnist */
    images_host = read_mnist_images(images_path.c_str(), n_images, n_rows, n_cols);
    labels_host = read_mnist_labels(labels_path.c_str(), n_labels, n_classes);
    n_features = n_rows * n_cols;


    model::nn_params_t params;
    params.batch_size = 32;    
    params.n_epochs = 5;

    auto x_batch = op::var<float>("x_batch", {params.batch_size, n_features}, {NONE,{}}, (use_gpu) ? DEVICE : HOST);

    auto input_layer = layer::input(x_batch);
    auto fc1 = layer::fullyconnected(input_layer->out(), 512);
    auto act1 = layer::activation(fc1->out(), layer::RELU);

    auto fc2 = layer::fullyconnected(act1->out(), 256);
    auto act2 = layer::activation(fc2->out(), layer::RELU);

    auto fc3 = layer::fullyconnected(act2->out(), n_classes);
    auto act3 = layer::activation(fc3->out(), layer::SOFTMAX);

    auto output = layer::output(act3->out());

    std::vector<layer::Layer<float> *> layers = {input_layer, fc1, act1, fc2, act2, fc3, act3, output};

    model::NeuralNetwork<float> model(layers, optimizer::CROSS_ENTROPY, optimizer::SGD, params);



    std::cout << "\n==== MNIST Interactive ====\n";
    do {
        std::cout << "\nOptions:\n";
        std::cout << "\t-1. EXIT\n";
        std::cout << "\t 0. Show Sample\n";
        std::cout << "\t 1. Train\n";
        std::cout << "\t 2. Predict\n";
        
        std::cout << "your option:  ";
        std::cin >> input;
        std::cout << "\n";

        switch (input) {
            case -1: break;
            case 0: show_sample(images_host, labels_host, n_rows, n_cols); break;
            case 1: train(model, images_host, labels_host); break;
            case 2: predict(model, images_host, labels_host, n_rows, n_cols); break;
            default:
                std::cout << "Invalid Option. Try again.\n"; break;
        }
    } while (input != -1);

    delete images_host;
    delete labels_host;
    delete output;

    magmadnn_finalize();
}


void show_sample(Tensor<float> *images, Tensor<float> *labels, uint32_t n_rows, uint32_t n_cols) {
    uint32_t image_idx = 0;

    std::cout << "Enter image index:  ";
    std::cin >> image_idx;
    std::cout << "\n";

    print_image(image_idx, images, labels, n_rows, n_cols);
}

void train(model::NeuralNetwork<float> &model, Tensor<float> *images, Tensor<float> *labels) {
    model::metric_t metrics;

    std::cout << "\n";
    model.fit(images, labels, metrics, true);
}

void predict(model::NeuralNetwork<float> &model, Tensor<float> *images, Tensor<float> *labels, uint32_t n_rows, uint32_t n_cols) {
    uint32_t image_index = 0, predicted_class;

    std::cout << "Enter image index:  ";
    std::cin >> image_index;
    std::cout << "\n";

    Tensor<float> sample ({n_rows * n_cols}, {NONE,{}}, images->get_memory_type());

    sample.copy_from(*images, image_index * sample.get_size(), sample.get_size());

    Tensor<float> *probas = model.predict(&sample);
    predicted_class = model.predict_class(&sample);

    std::cout << "Confidence(s):\n";
    for (unsigned int i = 0; i < labels->get_shape(1); i++) std::cout << std::setfill('-') << std::setw(11) << "-";
    std::cout << "\n|";
    for (unsigned int i = 0; i < labels->get_shape(1); i++) {
        std::cout << std::setfill(' ') << std::setw(5) << i << std::setw(3) << " " <<  " | ";
    }
    std::cout << "\n";
    for (unsigned int i = 0; i < labels->get_shape(1); i++) std::cout << std::setfill('-') << std::setw(11) << "-";
    std::cout << "\n|";
    for (unsigned int i = 0; i < labels->get_shape(1); i++) {
        std::cout << std::fixed << std::setprecision(6) << probas->get(i) << " | ";
    }
    std::cout << "\n";
    for (unsigned int i = 0; i < labels->get_shape(1); i++) std::cout << std::setfill('-') << std::setw(11) << "-";
    std::cout << "\n";

    std::cout << "predicted class = " << predicted_class << "\n";
    print_image(image_index, images, labels, n_rows, n_cols);
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