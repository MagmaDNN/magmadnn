/**
 * @file cnn_2d_cpu.cpp
 */
#include <cstdint>
#include <cstdio>
#include <vector>

#include "magmadnn.h"

using namespace magmadnn;

/* these are used for reading in the MNIST data set -- found at http://yann.lecun.com/exdb/mnist/ */
Tensor<float> *read_mnist_images(const char *file_name, uint32_t &n_images, uint32_t &n_rows, uint32_t &n_cols);
Tensor<float> *read_mnist_labels(const char *file_name, uint32_t &n_labels, uint32_t n_classes);
void print_image(uint32_t image_idx, Tensor<float> *images, Tensor<float> *labels, uint32_t n_rows, uint32_t n_cols);

int main(int argc, char **argv) {
    magmadnn_init();

#ifdef _OPENMP
    printf("omp enabled.\n");
#endif

    Tensor<float> *images_host, *labels_host;
    uint32_t n_images, n_rows, n_cols, n_labels, n_classes = 10;
    memory_t training_memory_type;

    /* these functions read-in and return tensors holding the mnist data set
        to use them, please change the string to the path to your local copy of the mnist dataset.
        it can be downloaded from http://yann.lecun.com/exdb/mnist */
    images_host = read_mnist_images("/opt/data/mnist/train-images-idx3-ubyte", n_images, n_rows, n_cols);
    labels_host = read_mnist_labels("/opt/data/mnist/train-labels-idx1-ubyte", n_labels, n_classes);
    printf("loaded images\n");
    if (images_host == NULL || labels_host == NULL) {
        return 1;
    }

    if (argc == 2) {
        print_image(std::atoi(argv[1]), images_host, labels_host, n_rows, n_cols);
    }

    model::nn_params_t params;
    params.batch_size = 128;
    params.n_epochs = 5;
    params.learning_rate = 0.05;

    // #if defined(USE_GPU)
    //     training_memory_type = DEVICE;
    // #else
    training_memory_type = HOST;
    // #endif

    auto x_batch = op::var<float>("x_batch", {params.batch_size, 1, n_rows, n_cols}, {NONE, {}}, training_memory_type);

    auto input = layer::input<float>(x_batch);

    auto conv2d1 = layer::conv2d<float>(input->out(), {5, 5}, 12, {0, 0}, {2, 2}, {1, 1}, true, false);
    auto act1 = layer::activation<float>(conv2d1->out(), layer::RELU);
    auto pool1 = layer::pooling<float>(act1->out(), {2, 2}, {0, 0}, {2, 2}, {1, 1}, MAX_POOL);
    auto dropout1 = layer::dropout<float>(pool1->out(), 0.25);
    auto flatten = layer::flatten<float>(dropout1->out());

    auto fc1 = layer::fullyconnected<float>(flatten->out(), 128, true);
    auto act2 = layer::activation<float>(fc1->out(), layer::RELU);
    auto fc2 = layer::fullyconnected<float>(act2->out(), n_classes, false);
    auto act3 = layer::activation<float>(fc2->out(), layer::SOFTMAX);

    auto output = layer::output<float>(act3->out());

    std::vector<layer::Layer<float> *> layers = {input, conv2d1, act1, pool1, dropout1, flatten,
                                                 fc1,   act2,    fc2,  act3,  output};

    // std::vector<layer::Layer<float> *> layers = {input, conv2d1, act1, dropout1, flatten, fc1, act2, fc2, act3,
    // output};

    model::NeuralNetwork<float> model(layers, optimizer::CROSS_ENTROPY, optimizer::SGD, params);

    model::metric_t metrics;

    model.fit(images_host, labels_host, metrics, true);

    delete images_host;
    delete labels_host;
    delete output;

    magmadnn_finalize();

    return 0;
}

#define FREAD_CHECK(res, nmemb)           \
    if ((res) != (nmemb)) {               \
        fprintf(stderr, "fread fail.\n"); \
        return NULL;                      \
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
    data = new Tensor<float>({n_images, 1, n_rows, n_cols}, {NONE, {}}, HOST);

    for (uint32_t i = 0; i < n_images; i++) {
        FREAD_CHECK(fread(bytes, sizeof(char), n_rows * n_cols, file), n_rows * n_cols);

        for (uint32_t r = 0; r < n_rows; r++) {
            for (uint32_t c = 0; c < n_cols; c++) {
                val = bytes[r * n_cols + c];

                data->set(i * n_rows * n_cols + r * n_cols + c, (val / 128.0f) - 1.0f);
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
    labels = new Tensor<float>({n_labels, n_classes}, {ZERO, {}}, HOST);

    printf("finished reading labels.\n");

    for (unsigned int i = 0; i < n_labels; i++) {
        FREAD_CHECK(fread(&val, sizeof(char), 1, file), 1);

        labels->set(i * n_classes + val, 1.0f);
    }

    fclose(file);

    return labels;
}

void print_image(uint32_t image_idx, Tensor<float> *images, Tensor<float> *labels, uint32_t n_rows, uint32_t n_cols) {
    uint8_t label = 0;
    uint32_t n_classes = labels->get_shape(1);

    /* assign the label */
    for (uint32_t i = 0; i < n_classes; i++) {
        if (std::fabs(labels->get(image_idx * n_classes + i) - 1.0f) <= 1E-8) {
            label = i;
            break;
        }
    }

    printf("Image[%u] is digit %u:\n", image_idx, label);

    for (unsigned int r = 0; r < n_rows; r++) {
        for (unsigned int c = 0; c < n_cols; c++) {
            printf("%3u ", (uint8_t)((images->get(image_idx * n_rows * n_cols + r * n_cols + c) + 1.0f) * 128.0f));
        }
        printf("\n");
    }
}

#undef FREAD_CHECK